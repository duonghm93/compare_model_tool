from pyspark.sql import SparkSession
import pickle
import os


def get_df_train(spark):
    df = spark.read.csv(
        ['gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid46/20180423_20180429',
         'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid46/20180416_20180422'],
        header=True, inferSchema=True)
    return df


def get_df_test(spark):
    df = spark.read.csv(
        'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid46/20180430_20180506',
        header=True, inferSchema=True
    )
    return df


def create_hist(df, field, bins):
    return df.select(field).rdd.flatMap(lambda x: x).histogram(buckets=bins)


def create_and_export_hist(df, field, bins, local_fn, gcs_folder):
    hist = create_hist(df, field, bins)
    pickle.dump(hist, open(local_fn, 'wb'))
    gcs_location = os.path.join(gcs_folder, local_fn).replace('\\','/')
    os.system('gsutil cp -r %s %s' %(local_fn, gcs_location))


if __name__ == '__main__':
    spark = SparkSession.builder.appName("check_data").getOrCreate()
    sponsor_ids = [37, 46, 55, 67, 70]
    for sponsor_id in sponsor_ids:
        # gcs_hist_export_folder = 'gs://reemo/models/dev/test_duong/test_j14/hist/spid%d/train/'%sponsor_id
        gcs_hist_export_folder = 'gs://reemo/models/dev/test_duong/test_j14/hist/spid%d/test/' % sponsor_id
        # df = spark.read.csv(
        #     ['gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180423_20180429'%sponsor_id,
        #      'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180416_20180422'%sponsor_id],
        #     header=True, inferSchema=True)
        df = spark.read.csv(
            'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/20180430_20180506' % sponsor_id,
            header=True, inferSchema=True)
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