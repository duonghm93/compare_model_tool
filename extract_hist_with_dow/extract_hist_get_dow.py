from pyspark.sql import SparkSession
import pickle
import os


def create_hist(df, field, bins):
    return df.select(field).rdd.flatMap(lambda x: x).histogram(buckets=bins)


def create_and_export_hist(df, field, bins, local_fn, gcs_folder):
    hist = create_hist(df, field, bins)
    pickle.dump(hist, open(local_fn, 'wb'))
    gcs_location = os.path.join(gcs_folder, local_fn).replace('\\','/')
    os.system('gsutil cp -r %s %s' %(local_fn, gcs_location))


def preproc_v2p1(df):
    df = _drop_ng_rows(df)
    df = _fill_na_v2p2(df)
    if df.count() != df.na.drop().count():
        print('There are NULL values in the processed data. Something wrong.')
        exit(1)
    return df


def _drop_ng_rows(df):
    # not_rtg = df['recency'] < 1e-10  # a bug; this line should be as below
    not_rtg = df['elapsed_time_rt'] == -1
    null_gender = df['prob_man'].isNull()
    null_ctr_user = df['ctr_user'].isNull() | (df['ctr_user'] < 1e-10)
    unknown_slot = df['ctr_slot'].isNull()
    mask_drop = (not_rtg & null_gender & null_ctr_user) | unknown_slot
    return df.filter(~mask_drop)


def _fill_na_v2p2(df):
    default_hr_stats_value = get_average_stat_value(range_size=7, stat_field_name='hr_range', ratio_field_name='ratio')
    default_dow_stats_value = get_average_stat_value(range_size=7, stat_field_name='dow_range', ratio_field_name='ratio')
    df = df.na.fill({
        'prob_man': -1.0, 'ctr_user': -1.0,
        'prob_man_stats': -1.0, 'uu_ratio': -1.0, 'ctr_user_avg': -1.0,
        'dow_stats': default_dow_stats_value, 'hour_stats': default_hr_stats_value,
        'slot_sponsor_rt_stats': 0.0, 'slot_sponsor_cv_stats': 0.0,
        'sponsor_rt_probman_stats': -1.0, 'sponsor_cv_probman_stats': -1.0
    })
    # rec and inview_rec of ReeMo data: 0 --> -1
    # "pabs" is needed because those of AkaNe data are -2 (and the vaue of -2 should keep)
    # df = df.withColumn('recency', F.when(pabs(df['recency']) < 1e-10, -1).otherwise(df['recency']))\
        # .withColumn('inview_recency', F.when(pabs(df['inview_recency']) < 1e-10, -1).otherwise(df['inview_recency']))
    return df


def get_average_stat_value(range_size, stat_field_name , ratio_field_name='ratio'):
    result = []
    if range_size > 0:
        average_value = 1.0 / range_size
        for range_id in range(0,range_size):
            result.append('{"%s":%d, "%s":%f}'%(stat_field_name, range_id, ratio_field_name, average_value))
    str_result = ','.join(result)
    return '[' + str_result + ']'


def extract_column_to_feature_v2p2(df):
    df = df.withColumnRenamed('hour_stats', 'hour_stats_org')
    for hr_range in range(0, 7):
        udf_hr_stat = UserDefinedFunction(lambda row: get_ratio_hr_stat(row, hr_range), DoubleType())
        df = df.withColumn(('hour_stat_' + str(hr_range)), udf_hr_stat(col('hour_stats_org')))

    df = df.withColumnRenamed('dow_stats', 'dow_stats_org')
    for dow_range in range(0, 7):
        udf_dow_stat = UserDefinedFunction(lambda row: get_ratio_dow_stat(row, dow_range), DoubleType())
        df = df.withColumn(('dow_stat_' + str(dow_range)), udf_dow_stat(col('dow_stats_org')))
    return df


if __name__ == '__main__':
    spark = SparkSession.builder.appName("check_data").getOrCreate()
    sponsor_ids = [37, 46, 55, 67, 70]
    for sponsor_id in sponsor_ids:
        gcs_hist_export_folder = 'gs://reemo/models/dev/test_duong/test_j14/hist/spid%d/train/'%sponsor_id
        # gcs_hist_export_folder = 'gs://reemo/models/dev/test_duong/test_j14/hist/spid%d/test/' % sponsor_id
        df = spark.read.csv(
            ['gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180423_20180429'%sponsor_id,
             'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180416_20180422'%sponsor_id],
            header=True, inferSchema=True)
        df = preproc_v2p1(df)

        # df = spark.read.csv(
        #     'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180430_20180506' % sponsor_id,
        #     header=True, inferSchema=True)
        df_reemo = df.filter(df.dsp_id == 1).cache()
        df_reemo_isclick = df_reemo.filter(df_reemo.is_click == 1).cache()
        df_reemo_nonclick = df_reemo.filter(df_reemo.is_click == 0).cache()

        df_akane = df.filter(df.dsp_id == 2).cache()
        df_akane_isclick = df_akane.filter(df_akane.is_click == 1).cache()
        df_akane_nonclick = df_akane.filter(df_akane.is_click == 0).cache()

        # fields = ['uu_ratio', 'ctr_user_avg', 'prob_man_stats', 'slot_sponsor_rt_stats', 'slot_sponsor_cv_stats']
        fields = ['prob_man', 'ctr_sp_slot', 'ctr_slot', 'ctr_user', 'iv_ctr_sp_slot']
        bins = [x*0.00001 for x in range(100001)]
        for field in fields:
            create_and_export_hist(df_reemo_isclick, field, bins, '%s_reemo_isclick.hist'%field, gcs_hist_export_folder)
            create_and_export_hist(df_reemo_nonclick, field, bins, '%s_reemo_nonclick.hist'%field, gcs_hist_export_folder)
            create_and_export_hist(df_akane_isclick, field, bins, '%s_akane_isclick.hist'%field, gcs_hist_export_folder)
            create_and_export_hist(df_akane_nonclick, field, bins, '%s_akane_nonclick.hist'%field, gcs_hist_export_folder)