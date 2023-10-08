Сначала запускем кластер. Для этого зададим конфигурацию, в файле spark_env.sh укажем значения:

- SPARK_WORKER_CORES=2
- SPARK_WORKER_INSTANCES=2
- SPARK_WORKER_MEMORY=2g

Это означает что при запуске кластера будут использоваться 2 worker, для каждого доступно 2 ядра cpu и 2 Гб оперативной памяти.

После этого, сначала запускаем мастера, после чего подключаем к нему воркеры:

```console
sudo sbin/start-master.sh
sudo sbin/start-worker.sh spark://lightkeeper:7077
```
Далее в jupyter noteebook подключаемся к данному кластеру.
