import os
from pyspark.sql import SparkSession


def get_slot_distribution(df):
    df_slot_count = df.groupBy('ssp_id', 'slot_id').count()
    return df_slot_count.toPandas()


def extract_slot_distribution(df, local_fn, gcs_folder):
    df_slot = get_slot_distribution(df)
    df_slot.to_csv(local_fn, index = None)
    gcs_location = os.path.join(gcs_folder, local_fn).replace('\\','/')
    os.system('gsutil cp -r %s %s' %(local_fn, gcs_location))


if __name__ == '__main__':
    spark = SparkSession.builder.appName("check_data").getOrCreate()
    sponsor_ids = [37, 46, 55, 67, 70]
    data_periods = ['20180430_20180506']
    gcs_base_export_folder = 'gs://reemo/models/dev/test_duong/test_j14/ctr_model/slot_distribution/'
    # gcs_base_export_folder = 'C:/Users/hoang/Desktop/check_25f_abnormal/ctr_model/slot_distribution/'

    for sponsor_id in sponsor_ids:
        for period in data_periods:
            df = spark.read.csv(
                'gs://reemo/models/dev/test_duong/test_j14/ctr_model/training_data/spid%d/%s'%(sponsor_id, period),
                header=True, inferSchema=True)
            # df = spark.read.csv(
            #     'C:/Users/hoang/Desktop/check_25f_abnormal/ctr_model/training_data/spid%d/%s' % (sponsor_id, period),
            #     header=True, inferSchema=True)
            df_reemo = df.filter(df.dsp_id == 1).cache()
            df_akane = df.filter(df.dsp_id == 2).cache()

            df_reemo_isclick = df_reemo.filter(df_reemo.is_click == 1).cache()
            df_reemo_nonclick = df_reemo.filter(df_reemo.is_click == 0).cache()
            df_akane_isclick = df_akane.filter(df_akane.is_click == 1).cache()
            df_akane_nonclick = df_akane.filter(df_akane.is_click == 0).cache()

            gcs_export_folder = os.path.join(gcs_base_export_folder, period, 'spid%d'%sponsor_id).replace('\\','/')

            extract_slot_distribution(df_reemo_isclick, 'reemo_isclick.csv', gcs_export_folder)
            extract_slot_distribution(df_reemo_nonclick, 'reemo_nonclick.csv', gcs_export_folder)
            extract_slot_distribution(df_akane_isclick, 'akane_isclick.csv', gcs_export_folder)
            extract_slot_distribution(df_akane_nonclick, 'akane_nonclick.csv', gcs_export_folder)