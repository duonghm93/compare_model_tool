from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import json


# ======== Version 2.2.1 =========
def preproc_v2p2_1(df):
    return preproc_v2p2(df)


def get_names_features_v2p2_1(df):
    # the last five columns are 'sponsor_id, ssp_id, slot_id, dsp_id, is_click'
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
        , 'prob_man_stats'
        , 'uu_ratio'
        , 'ctr_user_avg'
        , 'hour_stat_0', 'hour_stat_1', 'hour_stat_2', 'hour_stat_3', 'hour_stat_4', 'hour_stat_5', 'hour_stat_6'
        , 'weekday_stat', 'weekend_stat'
        , 'slot_sponsor_rt_stats', 'slot_sponsor_cv_stats'
        , 'sponsor_rt_probman_stats', 'sponsor_cv_probman_stats'
    ]
    return f_names


def extract_column_to_feature_v2p2_1(df):
    df = extract_column_to_feature_v2p2(df)
    df = df.withColumn('weekday_stat', sum(df[col] for col in ['dow_stat_%d'%i for i in range(5)]))
    df = df.withColumn('weekend_stat', sum(df[col] for col in ['dow_stat_%d' % i for i in range(5,7)]))
    return df


# ======== Version 2.2 =========
def _drop_ng_rows_v2p2(df):
    return _drop_ng_rows(df)


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


def preproc_v2p2(df):
    df = _drop_ng_rows_v2p2(df)
    df = _fill_na_v2p2(df)
    if df.count() != df.na.drop().count():
        print('There are NULL values in the processed data. Something wrong.')
        exit(1)
    return df


def get_names_features_v2p2(df):
    # the last five columns are 'sponsor_id, ssp_id, slot_id, dsp_id, is_click'
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
        , 'prob_man_stats'
        , 'uu_ratio'
        , 'ctr_user_avg'
        , 'hour_stat_0', 'hour_stat_1', 'hour_stat_2', 'hour_stat_3', 'hour_stat_4', 'hour_stat_5', 'hour_stat_6'
        , 'dow_stat_0', 'dow_stat_1', 'dow_stat_2', 'dow_stat_3', 'dow_stat_4', 'dow_stat_5', 'dow_stat_6'
        , 'slot_sponsor_rt_stats', 'slot_sponsor_cv_stats'
        , 'sponsor_rt_probman_stats', 'sponsor_cv_probman_stats'
    ]
    return f_names


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


def get_ratio_hr_stat(hr_stat_str, hr_range_id):
    ratio = 0.0
    hr_stat_str = '[' + hr_stat_str[1:-1] + ']'
    hr_stats = json.loads(hr_stat_str)
    ratios = [x['ratio'] for x in hr_stats if x['hr_range'] == hr_range_id]
    if len(ratios) > 0:
        ratio = float(ratios[0])
    return ratio


def get_ratio_dow_stat(dow_stat_str, dow_range_id):
    ratio = 0.0
    dow_stat_str = '[' + dow_stat_str[1:-1] + ']'
    dow_stats = json.loads(dow_stat_str)
    ratios = [x['ratio'] for x in dow_stats if x['dow_range'] == dow_range_id]
    if len(ratios) > 0:
        ratio = float(ratios[0])
    return ratio


def get_average_stat_value(range_size, stat_field_name , ratio_field_name='ratio'):
    result = []
    if range_size > 0:
        average_value = 1.0 / range_size
        for range_id in range(0,range_size):
            result.append('{"%s":%d, "%s":%f}'%(stat_field_name, range_id, ratio_field_name, average_value))
    str_result = ','.join(result)
    return '[' + str_result + ']'

# ======== Version 2.03 ========
def preproc_v2_03(df):
    return preproc(df)


def get_names_features_v2_03(df):
    return get_names_features(df)


# ======== Version 2.02 ========
def preproc_v2_02(df):
    return preproc(df)


def get_names_features_v2_02(df):
    return get_names_features(df)


# ======== Base preproc ==========
def _drop_ng_rows(df):
    # not_rtg = df['recency'] < 1e-10  # a bug; this line should be as below
    not_rtg = df['elapsed_time_rt'] == -1
    null_gender = df['prob_man'].isNull()
    null_ctr_user = df['ctr_user'].isNull() | (df['ctr_user'] < 1e-10)
    unknown_slot = df['ctr_slot'].isNull()
    mask_drop = (not_rtg & null_gender & null_ctr_user) | unknown_slot
    return df.filter(~mask_drop)


def _fill_na(df):
    df = df.na.fill({'prob_man': -1.0, 'ctr_user': -1.0})
    # rec and inview_rec of ReeMo data: 0 --> -1
    # "pabs" is needed because those of AkaNe data are -2 (and the vaue of -2 should keep)
    # df = df.withColumn('recency', F.when(pabs(df['recency']) < 1e-10, -1).otherwise(df['recency']))\
        # .withColumn('inview_recency', F.when(pabs(df['inview_recency']) < 1e-10, -1).otherwise(df['inview_recency']))
    return df


def preproc(df):
    df = _drop_ng_rows(df)
    df = _fill_na(df)
    if df.count() != df.na.drop().count():
        print('There are NULL values in the processed data. Something wrong.')
        exit(1)
    return df


def get_names_features(df):
    # the last five columns are 'sponsor_id, ssp_id, slot_id, dsp_id, is_click'
    f_names = df.columns[:-5]
    return f_names


def extract_column_to_feature(df):
    return df


def get_names_features_v2p1(df):
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
    ]
    return f_names


def preproc_v2p1(df):
    df = _drop_ng_rows(df)
    df = _fill_na_v2p2(df)
    if df.count() != df.na.drop().count():
        print('There are NULL values in the processed data. Something wrong.')
        exit(1)
    return df


def get_names_features_v2p1p2(df):
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
        , 'ctr_user_avg', 'uu_ratio', 'prob_man_stats'
        , 'slot_sponsor_rt_stats', 'slot_sponsor_cv_stats'
    ]
    return f_names


def get_names_features_v2p1p2_22f_remove_slot_feature(df):
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
        , 'slot_sponsor_rt_stats', 'slot_sponsor_cv_stats'
    ]
    return f_names


def get_names_features_v2p1p2_23f_remove_sp_slot_feature(df):
    f_names = [
        'hr'
        , 'dow'
        , 'fq'
        , 'recency'
        , 'inview_fq'
        , 'inview_recency'
        , 'elapsed_time_rt'
        , 'is_rt_any'
        , 'is_same_domain'
        , 'creative_type'
        , 'os'
        , 'prob_man'
        , 'ctr_user'
        , 'inview_ratio'
        , 'ctr_slot'
        , 'iv_ctr_slot'
        , 'slot_category'
        , 'slot_site_type'
        , 'ctr_sp_slot'
        , 'iv_ctr_sp_slot'
        , 'ctr_user_avg', 'uu_ratio', 'prob_man_stats'
    ]
    return f_names
