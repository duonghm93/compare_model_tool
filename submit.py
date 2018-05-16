import os


def create_cluster(cluster_name):
    cluster_option = '--image-version 1.1 --zone asia-east1-b ' \
                     '--num-workers 6 --num-preemptible-workers 2 ' \
                     '--master-machine-type n1-highmem-8 --master-boot-disk-size 500 ' \
                     '--worker-machine-type n1-highmem-8 --worker-boot-disk-size 500 ' \
                     '--scopes https://www.googleapis.com/auth/cloud-platform ' \
                     '--project reemo-173606 --properties yarn:yarn.log-aggregation-enable=true '
    create_cluster_com = 'gcloud dataproc clusters create %s %s' % (cluster_name, cluster_option)
    os.system(create_cluster_com)


def delete_cluster(cluster_name):
    delete_cluster_com = 'gcloud -q dataproc clusters delete %s' %cluster_name
    os.system(delete_cluster_com)


if __name__ == '__main__':
    cluster_name = 'j14-seperate-predict-data'
    import_files = ['preproc_data.py']
    create_cluster(cluster_name)

    com_submit = 'compare_model.py'

    com = 'gcloud dataproc jobs submit pyspark --cluster %s' % cluster_name
    com += ' --properties'
    com += ' spark.executor.heartbeatInterval="4000s",spark.network.timeout="40000s"'
    com += ',spark.serializer=org.apache.spark.serializer.KryoSerializer'
    com += ' --py-files %s' % ','.join(import_files)
    com += ' %s' % (com_submit)
    print(com)
    os.system(com)

    delete_cluster(cluster_name)