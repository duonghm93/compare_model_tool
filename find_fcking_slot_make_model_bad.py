import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
import json
import preproc_data


# <editor-fold desc="Location manager">
MODEL_VERSION_INFO = {
    'v2p1_7_40':{
        # 'model_location': 'D:/test_compare_tool/',
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        'model_folder': '20180430_v2p1_7_40',
        'func_feature_names': preproc_data.get_names_features_v2p1,
        'func_extract_column': preproc_data.extract_column_to_feature
    },
    'v2p1p2_10_80_25f':{
        # 'model_location': 'D:/test_compare_tool/',
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        'model_folder': '20180430_v2p1p2_10_80_25f_remake',
        'func_feature_names': preproc_data.get_names_features_v2p1p2,
        'func_extract_column': preproc_data.extract_column_to_feature
    }
}
MODEL_FOLDER = 'ctr_model/model/'
# DEFAULT_GCS_FOLDER = 'D:/test_compare_tool/'
DEFAULT_GCS_FOLDER = 'gs://reemo/models/dev/test_duong/test_j14/'


def get_eval_data_location():
    return DEFAULT_GCS_FOLDER + 'ctr_model/training_data/spid46/20180430_20180506/'


def get_versions():
    return ['v2p1_7_40', 'v2p1p2_10_80_25f']


def get_hist_folder_location():
    return DEFAULT_GCS_FOLDER + 'hist_collection/hist_v2p1_rank5/20180430_20180506/'
# </editor-fold>


# <editor-fold desc="Process data">
def make_feature(df, version_name, convmap):
    for k in convmap:
        df = df.withColumn(k + '_' + version_name, categorical_conv(convmap[k]))


def get_convmap_dics(model_version, model_date):
    model_location = _get_model_version_folder(model_version)
    fn = 'convmap_ctr_model_%s.json' % model_date
    local_fn = '%s_%s' % (model_version, fn)
    gcs_fn = os.path.join(model_location, fn).replace('\\', '/')
    print(local_fn)
    print(gcs_fn)
    os.system('gsutil cp %s %s' % (gcs_fn, local_fn))
    return json.load(open(local_fn))


def convert_name_features(name_features, version, category_fields):
    return [field if field not in category_fields else field + '_' + version for field in name_features]


def _get_model_version_folder(model_version):
    version_infor = MODEL_VERSION_INFO[model_version]
    return os.path.join(
        version_infor['model_location'], MODEL_FOLDER, version_infor['model_folder']
    ).replace('\\', '/')


def get_model(model_version, spid, model_date):
    model_version_location = _get_model_version_folder(model_version)
    model_path = os.path.join(model_version_location, 'ctr_model_spid%d_%s' % (spid, model_date)).replace('\\', '/')
    return RandomForestClassificationModel.load(model_path)


def categorical_conv(convmap):
    f = lambda x: convmap.get(str(x), -1.0)
    return UserDefinedFunction(f, DoubleType())


def predict_with_multiple_version(df, versions, model_date, spid):
    columns = df.columns
    for version_name in versions:
        version_infor = MODEL_VERSION_INFO[version_name]
        convmaps = get_convmap_dics(version_name, model_date)
        for k in convmaps[str(spid)].keys():
            df = df.withColumn(k + '_' + version_name, categorical_conv(convmaps[str(spid)][k])(col(k)))
        name_features = version_infor['func_feature_names'](df)
        name_features = convert_name_features(name_features, version_name, list(convmaps[str(spid)]))
        df = VectorAssembler(inputCols=name_features, outputCol='features_%s' % version_name).transform(df)
    print(df.columns)
    predicted_list = []
    for version_name in versions:
        model = get_model(version_name, spid, model_date)
        prob_col_name = 'prob_%s' % version_name
        df = df.withColumn('features', col('features_%s' % version_name))
        df = model.transform(df).withColumn(prob_col_name, UserDefinedFunction(lambda x: x.tolist()[1], DoubleType())(
            col('probability')))
        predicted_list.append(version_name)
        df = df.select(columns + ['prob_%s' % v for v in predicted_list] + ['features_%s' % v for v in versions])
    df = df.select(columns + ['prob_%s' % v for v in versions])
    return df
#</editor-fold>


if __name__ == '__main__':
    spark = SparkSession.builder.appName("check_data").getOrCreate()
    versions = ['v2p1p2_10_80_25f']
    HIGH_PRED_THRES = 0.01

    eval_data_location = get_eval_data_location()
    df = spark.read.csv(eval_data_location, header=True, inferSchema=True)
    df = preproc_data.preproc_v2p1(df)

    df_akane = df.filter(df.dsp_id == 2)

    df_akane = predict_with_multiple_version(df_akane, versions, '20180430', 46)

    df_akane_isclick = df_akane.filter(df_akane.is_click == 1).cache()
    df_akane_isclick_high_pred = df_akane_isclick.filter(
        (col('prob_%s' % versions[0]) >= HIGH_PRED_THRES)
    ).cache()

    df_akane_nonclick = df_akane.filter(df_akane.is_click == 0).cache()
    df_akane_nonclick_high_pred = df_akane_nonclick.filter(
        (col('prob_%s' % versions[0]) >= HIGH_PRED_THRES)
    ).cache()

    gcs_result_folder = 'gs://reemo/models/dev/test_duong/test_j14/10_80_25f_analyze/'
    # gcs_result_folder = 'D:/test_compare_tool/10_80_25f_analyze/'

    df_slot_isclick_high = df_akane_isclick_high_pred.select('ssp_id', 'slot_id').groupBy('ssp_id', 'slot_id').count().cache()
    fn_isclick_high = 'spid46_10_80_25f_isclick_highctr.csv'
    df_slot_isclick_high.toPandas().to_csv('spid46_10_80_25f_isclick_highctr.csv', index=None)
    os.system('gsutil cp %s %s' %(fn_isclick_high, gcs_result_folder+fn_isclick_high))

    df_slot_nonclick_high = df_akane_nonclick_high_pred.select('ssp_id', 'slot_id').groupBy('ssp_id', 'slot_id').count().cache()
    fn_nonclick_high = 'spid46_10_80_25f_nonclick_highctr.csv'
    df_slot_nonclick_high.toPandas().to_csv('spid46_10_80_25f_nonclick_highctr.csv', index=None)
    os.system('gsutil cp %s %s' % (fn_nonclick_high, gcs_result_folder + fn_nonclick_high))
