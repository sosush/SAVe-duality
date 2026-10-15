import os
import sys
import time
import subprocess

# Enforce working Java 21 JDK on macOS
JAVA_21_PATH = "/Users/sohinibanerjee/Library/Java/JavaVirtualMachines/jbr-21.0.11/Contents/Home"
if os.path.exists(JAVA_21_PATH):
    os.environ["JAVA_HOME"] = JAVA_21_PATH
    os.environ["PATH"] = f"{JAVA_21_PATH}/bin:" + os.environ.get("PATH", "")

# Load Spark-Kafka Connector
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, expr
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, FloatType

spark = SparkSession.builder \
    .appName("SAVe-Duality-Spark-NOC") \
    .master("local[*]") \
    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

LAST_NOTIF_TIME = 0.0

def send_mac_notification(title, message):
    global LAST_NOTIF_TIME
    now = time.time()
    if now - LAST_NOTIF_TIME > 3.0:
        cmd = f'osascript -e \'display notification "{message}" with title "{title}" sound name "Basso"\''
        subprocess.Popen(cmd, shell=True)
        LAST_NOTIF_TIME = now

video_schema = StructType([
    StructField("camera_id", StringType()),
    StructField("pts", LongType()),
    StructField("frame_idx", LongType()),
    StructField("frame_b64", StringType())
])

audio_schema = StructType([
    StructField("camera_id", StringType()),
    StructField("pts", LongType()),
    StructField("frame_idx", LongType()),
    StructField("audio_chunk", ArrayType(FloatType()))
])

# Stream from Kafka
video_raw = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092").option("subscribe", "video-stream").load()
audio_raw = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:9092").option("subscribe", "audio-stream").load()

v_df = video_raw.select(from_json(col("value").cast("string"), video_schema).alias("data")) \
    .select(col("data.camera_id").alias("v_cam"), col("data.pts").alias("v_pts"), col("data.frame_idx").alias("v_idx")) \
    .withColumn("v_time", (col("v_pts") / 1000).cast("timestamp")) \
    .withWatermark("v_time", "10 seconds")

a_df = audio_raw.select(from_json(col("value").cast("string"), audio_schema).alias("data")) \
    .select(col("data.camera_id").alias("a_cam"), col("data.pts").alias("a_pts"), col("data.frame_idx").alias("a_idx")) \
    .withColumn("a_time", (col("a_pts") / 1000).cast("timestamp")) \
    .withWatermark("a_time", "10 seconds")

# Sub-second Cross-Modal Temporal Join
synced = v_df.join(
    a_df,
    expr("v_cam = a_cam AND v_idx = a_idx AND a_time >= v_time - interval 5 seconds AND a_time <= v_time + interval 5 seconds")
)

def process_batch(batch_df, batch_id):
    records = batch_df.collect()
    if not records:
        return

    print("\n" + "="*85)
    print(" 🛡️  APACHE SPARK STRUCTURED STREAMING: SUB-SECOND SYNCHRONIZATION MONITOR")
    print("="*85)

    for row in records:
        cam_id = row["v_cam"]
        v_pts = row["v_pts"]
        a_pts = row["a_pts"]
        skew = abs(v_pts - a_pts)

        if skew > 150:
            status = "🚨 TAMPERING DETECTED [AUDIO-VISUAL INCOHERENCE / DESYNC]"
            send_mac_notification("SPARK SECURITY ALERT", f"Skew of {skew}ms detected on {cam_id}")
            os.makedirs("data", exist_ok=True)
            with open("data/iceberg_anomalies.log", "a") as f:
                f.write(f'{{"camera": "{cam_id}", "skew_ms": {skew}, "flag": "SPARK_INCOHERENCE"}}\n')
        else:
            status = "✅ SYNCHRONIZED STREAM"

        print(f"[{cam_id:20s}] PTS Skew: {skew:04d}ms | {status}")
    print("="*85 + "\n")

print("[*] Spark NOC Engine running. Listening to Kafka broker on :9092...")
query = synced.writeStream.foreachBatch(process_batch).start()
query.awaitTermination()