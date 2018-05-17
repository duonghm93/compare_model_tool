import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
import json
import preproc_data

# --- PARAM ---
MAX_DATA_ROW = 1e6

TRAINING_DATA_FOLDER = 'ctr_model/training_data/'
MODEL_FOLDER = 'ctr_model/model/'
DEFAULT_MODEL_LOCATION = 'gs://reemo/models/dev/test_duong/test_j14/'
# DEFAULT_MODEL_LOCATION = 'D:/test_compare_tool/'
PREDICT_FOLDER = 'ctr_model/predict/'

MODEL_VERSION_INFO = {
    'v2p1_7_40':{
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        # 'model_location': 'D:/test_compare_tool/',
        'model_folder': '20180430_v2p1_7_40',
        'func_feature_names': preproc_data.get_names_features_v2p1,
        'func_extract_column': preproc_data.extract_column_to_feature
    },
    'v2p1p2_8_40_25f':{
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        # 'model_location': 'D:/test_compare_tool/',
        'model_folder': '20180430_v2p1p2_8_40_25f',
        'func_feature_names': preproc_data.get_names_features_v2p1p2,
        'func_extract_column': preproc_data.extract_column_to_feature
    }
}


def get_model_versions():
    return ['v2p1_7_40', 'v2p1p2_8_40_25f']


def get_sponsor_ids():
    return [37, 46]


def get_model_date():
    return '20180430'


def get_evaluate_periods():
    return ['20180430_20180506']
# --------------


# --- SUPPORT FUNCTION ---
def get_training_data_folder():
    return os.path.join(DEFAULT_MODEL_LOCATION, TRAINING_DATA_FOLDER).replace('\\','/')


def get_data_location(spid, periods):
    training_folder = get_training_data_folder()
    result = [os.path.join(training_folder, 'spid'+str(spid), period).replace('\\','/') for period in periods]
    return result


def get_spark():
    spark = SparkSession.builder \
        .appName("check_data").getOrCreate()
    return spark


def get_data(spid, periods):
    data_location = get_data_location(spid, periods)
    spark = get_spark()
    df = spark.read.csv(data_location, header=True, inferSchema=True)
    return df


def _get_model_version_folder(model_version):
    version_infor = MODEL_VERSION_INFO[model_version]
    return os.path.join(
        version_infor['model_location'], MODEL_FOLDER, version_infor['model_folder']
    ).replace('\\','/')


def get_prediction_folder():
    return os.path.join(DEFAULT_MODEL_LOCATION, PREDICT_FOLDER).replace('\\','/')
# -----------------------


# --- ABSTRACT FUNCTION ------
def get_model(model_version, spid, model_date):
    model_version_location = _get_model_version_folder(model_version)
    model_path = os.path.join(model_version_location, 'ctr_model_spid%d_%s'%(spid, model_date)).replace('\\','/')
    return RandomForestClassificationModel.load(model_path)


def get_convmap_dics(model_version, model_date):
    model_location = _get_model_version_folder(model_version)
    fn = 'convmap_ctr_model_%s.json' % model_date
    local_fn = '%s_%s' % (model_version, fn)
    gcs_fn = os.path.join(model_location, fn).replace('\\','/')
    os.system('gsutil cp %s %s' %(gcs_fn, local_fn))
    return json.load(open(local_fn))


def categorical_conv(convmap):
    f = lambda x: convmap.get(str(x), -1.0)
    return UserDefinedFunction(f, DoubleType())


def convert_name_features(name_features, version, category_fields):
    return [field if field not in category_fields else field+'_'+version for field in name_features]
# ----------------------------


# --- Process data ---
def make_feature(df, version_name, convmap):
    for k in convmap:
        df = df.withColumn(k + '_' + version_name, categorical_conv(convmap[k]))


def predict_with_multiple_version(df, versions, model_date, spid):
    for version_name in versions:
        version_infor = MODEL_VERSION_INFO[version_name]
        convmaps = get_convmap_dics(version_name, model_date)
        for k in convmaps[str(spid)].keys():
            df = df.withColumn(k + '_' + version_name, categorical_conv(convmaps[str(spid)][k])(col(k)))
        name_features = version_infor['func_feature_names'](df)
        name_features = convert_name_features(name_features, version_name, list(convmaps[str(spid)]))
        df = VectorAssembler(inputCols=name_features, outputCol='features_%s'%version_name).transform(df)
    print(df.columns)
    predicted_list = []
    for version_name in versions:
        model = get_model(version_name, spid, model_date)
        prob_col_name = 'prob_%s'%version_name
        df = df.withColumn('features', col('features_%s'%version_name))
        df = model.transform(df).withColumn(prob_col_name, UserDefinedFunction(lambda x: x.tolist()[1], DoubleType())(col('probability')))
        predicted_list.append(version_name)
        df = df.select(['is_click', 'dsp_id'] + ['prob_%s'%v for v in predicted_list] + ['features_%s'%v for v in versions])
    df = df.select(['is_click', 'dsp_id'] + ['prob_%s'%v for v in versions])
    return df


# --------------------

# --- Main function ----
if __name__ == '__main__':
    sponsor_ids = get_sponsor_ids()
    model_date = get_model_date()
    test_periods = get_evaluate_periods()
    versions = get_model_versions()

    for sponsor_id in sponsor_ids:
        df = get_data(sponsor_id, test_periods)
        df = preproc_data.preproc_v2p1(df)
        df.cache()
        df_reemo = df.filter(df.dsp_id == 1)
        df_akane = df.filter(df.dsp_id == 2)

        df_reemo_click = df_reemo.filter(df_reemo.is_click == 1).cache()
        df_reemo_nonclick = df_reemo.filter(df_reemo.is_click == 0).cache()
        df_akane_click = df_akane.filter(df_akane.is_click == 1).cache()
        df_akane_nonclick = df_akane.filter(df_akane.is_click == 0).cache()

        reemo_click_size = df_reemo_click.count()
        reemo_nonclick_size = df_reemo_nonclick.count()
        akane_click_size = df_akane_click.count()
        akane_nonclick_size = df_akane_nonclick.count()

        print('Reemo: %d / %d = %.5f' %
              (reemo_click_size, reemo_nonclick_size, float(reemo_click_size)/reemo_nonclick_size))
        print('Akane: %d / %d = %.5f' %
              (akane_click_size, akane_nonclick_size, float(akane_click_size) / akane_nonclick_size))

        if reemo_click_size > 0:
            reemo_click_predict = predict_with_multiple_version(df_reemo_click, versions, model_date, sponsor_id)
            reemo_click_predict.write.csv(
                path=os.path.join(get_prediction_folder(), model_date, '%s_reemo_isclick'%sponsor_id).replace('\\','/'),
                header=True, mode='error')
        if akane_click_size > 0:
            akane_click_predict = predict_with_multiple_version(df_reemo_click, versions, model_date, sponsor_id)
            akane_click_predict.write.csv(
                path=os.path.join(get_prediction_folder(), model_date, '%s_akane_isclick'%sponsor_id).replace('\\','/'),
                header=True, mode='error'
            )
        if reemo_nonclick_size > 0:
            reemo_nonclick_predict = predict_with_multiple_version(df_reemo_nonclick, versions, model_date, sponsor_id)
            reemo_nonclick_predict.write.csv(
                path=os.path.join(get_prediction_folder(), model_date, '%s_reemo_nonclick'%sponsor_id).replace('\\','/'),
                header=True, mode='error'
            )
        if akane_nonclick_size > 0:
            akane_nonclick_predict = predict_with_multiple_version(df_akane_nonclick, versions, model_date, sponsor_id)
            akane_nonclick_predict.write.csv(
                path=os.path.join(get_prediction_folder(), model_date, '%s_akane_nonclick'%sponsor_id).replace('\\','/'),
                header=True, mode='error'
            )


