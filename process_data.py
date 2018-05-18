from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
import json
import os
import preproc_data


MODEL_VERSION_INFO = {
    'v2p1_7_40':{
        # 'model_location': 'C:/Users/hoang/Desktop/check_25f_abnormal/',
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        'model_folder': '20180430_v2p1_7_40',
        'func_feature_names': preproc_data.get_names_features_v2p1,
        'func_extract_column': preproc_data.extract_column_to_feature
    },
    'v2p1p2_10_80_25f':{
        # 'model_location': 'C:/Users/hoang/Desktop/check_25f_abnormal/',
        'model_location': 'gs://reemo/models/dev/test_duong/test_j14/',
        'model_folder': '20180430_v2p1p2_10_80_25f_remake',
        'func_feature_names': preproc_data.get_names_features_v2p1p2,
        'func_extract_column': preproc_data.extract_column_to_feature
    }
}
MODEL_FOLDER = 'ctr_model/model/'
# DEFAULT_GCS_FOLDER = 'C:/Users/hoang/Desktop/check_25f_abnormal/'
DEFAULT_GCS_FOLDER = 'gs://reemo/models/dev/test_duong/test_j14/'


def predict_with_multiple_version(df, versions, model_date, spid, versions_infor_dic):
    columns = df.columns
    for version_name in versions:
        version_infor = versions_infor_dic[version_name]
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



