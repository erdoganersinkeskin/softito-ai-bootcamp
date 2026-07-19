# ==============================================================================
# analyze_logs.py — Oyun Mağazası Loglarının PySpark ile Büyük Ölçekli Analizi
# ------------------------------------------------------------------------------
# ÖĞRENME NOTU (yönlendirme):
#   generate_logs.py ile üretilen ~1.000.000 satırlık log verisini PySpark
#   DataFrame API'si ve Spark SQL fonksiyonlarıyla analiz ediyoruz: durum
#   kodu dağılımı, en çok istek alan URL, ülke bazlı istatistikler, saatlik
#   dağılım, yavaş URL'ler, hata oranları, en aktif IP'ler, partition/veri
#   boyutu; ayrıca oyun kataloğu kullanmanın getirdiği bir bonus analiz
#   (en popüler tür/genre). Her analiz adımı için matplotlib grafiği
#   figures/ klasörüne kaydedilir.
#
#   Önce generate_logs.py'yi çalıştırıp logs/ klasörünü oluşturmanız gerekir.
# ==============================================================================

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum, desc, hour, when

os.makedirs("figures", exist_ok=True)

spark = SparkSession.builder \
    .appName("GameStoreLogAnalysis") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("\n=== 1. Veriyi Yükleme ===")
start = time.time()
df = spark.read.option("header", "true").csv("logs/*.csv")
df = df.withColumn("timestamp", col("timestamp").cast("bigint")) \
       .withColumn("response_time_ms", col("response_time_ms").cast("int")) \
       .withColumn("bytes_sent", col("bytes_sent").cast("int")) \
       .withColumn("status", col("status").cast("int"))
total = df.count()
print(f"Yüklenen satır: {total:,} | Süre: {time.time()-start:.2f}s")

print("\n=== 2. Şema ===")
df.printSchema()

print("\n=== 3. HTTP Durum Kodları Dağılımı ===")
start = time.time()
status_pd = df.groupBy("status").count().orderBy("status").toPandas()
print(status_pd.to_string(index=False))
print(f"Süre: {time.time()-start:.2f}s")

plt.figure(figsize=(8, 5))
plt.bar(status_pd["status"].astype(str), status_pd["count"], color="#3498DB")
plt.title("HTTP Durum Kodu Dağılımı")
plt.xlabel("Status Code")
plt.ylabel("İstek Sayısı")
plt.tight_layout()
plt.savefig("figures/status_codes.png", dpi=150)
plt.close()

print("\n=== 4. En Çok İstek Alan Oyun/Mağaza Sayfaları (Top 10) ===")
start = time.time()
top_urls_pd = df.groupBy("url").count().orderBy(desc("count")).limit(10).toPandas()
print(top_urls_pd.to_string(index=False))
print(f"Süre: {time.time()-start:.2f}s")

plt.figure(figsize=(9, 5))
plt.barh(top_urls_pd["url"][::-1], top_urls_pd["count"][::-1], color="#2ECC71")
plt.title("En Çok İstek Alan URL'ler (Top 10)")
plt.xlabel("İstek Sayısı")
plt.tight_layout()
plt.savefig("figures/top_urls.png", dpi=150)
plt.close()

print("\n=== 5. Ülkelere Göre İstek Sayısı ve Ortalama Yanıt Süresi ===")
start = time.time()
country_pd = df.groupBy("country").agg(
    count("*").alias("request_count"),
    avg("response_time_ms").alias("avg_response_ms"),
    avg("bytes_sent").alias("avg_bytes_sent")
).orderBy(desc("request_count")).toPandas()
print(country_pd.to_string(index=False))
print(f"Süre: {time.time()-start:.2f}s")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(country_pd["country"], country_pd["request_count"], color="#9B59B6")
axes[0].set_title("Ülkelere Göre İstek Sayısı")
axes[0].tick_params(axis="x", rotation=45)
axes[1].bar(country_pd["country"], country_pd["avg_response_ms"], color="#E67E22")
axes[1].set_title("Ülkelere Göre Ortalama Yanıt Süresi (ms)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figures/country_stats.png", dpi=150)
plt.close()

print("\n=== 6. Saatlik İstek Dağılımı (yoğun oyun saatlerini görmek için) ===")
start = time.time()
hourly_pd = df.withColumn("hour", hour((col("timestamp").cast("timestamp")))) \
  .groupBy("hour").count().orderBy("hour").toPandas()
print(hourly_pd.to_string(index=False))
print(f"Süre: {time.time()-start:.2f}s")

plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["count"], marker="o", color="#1ABC9C")
plt.fill_between(hourly_pd["hour"], hourly_pd["count"], alpha=0.2, color="#1ABC9C")
plt.title("Saatlik İstek Dağılımı")
plt.xlabel("Saat")
plt.ylabel("İstek Sayısı")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("figures/hourly_distribution.png", dpi=150)
plt.close()

print("\n=== 7. Yavaş Uç Noktalar (ortalama > 2000ms, Top 10) ===")
start = time.time()
df.groupBy("url").agg(
    avg("response_time_ms").alias("avg_response"),
    count("*").alias("count")
).filter(col("avg_response") > 2000).orderBy(desc("avg_response")).show(10, truncate=False)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 8. HTTP Hata Oranları (4xx/5xx, Top 10) ===")
start = time.time()
df.withColumn("is_error", when(col("status") >= 400, 1).otherwise(0)) \
  .groupBy("url").agg(
      count("*").alias("total"),
      sum("is_error").alias("errors")
  ).withColumn("error_rate_pct", (col("errors") / col("total") * 100)) \
   .orderBy(desc("error_rate_pct")).show(10, truncate=False)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 9. En Aktif Kullanıcılar / IP'ler (Top 10) ===")
start = time.time()
df.groupBy("ip").count().orderBy(desc("count")).show(10)
print(f"Süre: {time.time()-start:.2f}s")

print("\n=== 10. Partition Sayısı & Veri Boyutu ===")
total_bytes = df.select(sum("bytes_sent")).collect()[0][0]
print(f"Toplam bayt gönderimi: {total_bytes:,}")
print(f"Partition sayısı: {df.rdd.getNumPartitions()}")

# ------------------------------------------------------------------------------
# BONUS ANALİZ (oyun kataloğu kullanmanın getirdiği ek imkan): appid/genre
# kolonları sayesinde hangi TÜR oyunların en çok trafik
# aldığını görebiliyoruz. "genre" boş olan satırlar (oyuna özel olmayan genel
# mağaza istekleri, ör. /login, /search) hariç tutuluyor.
# ------------------------------------------------------------------------------
print("\n=== 11. [BONUS] En Popüler Oyun Türleri (Genre, Top 10) ===")
start = time.time()
genre_pd = df.filter((col("genre").isNotNull()) & (col("genre") != "")) \
  .groupBy("genre").count().orderBy(desc("count")).limit(10).toPandas()
print(genre_pd.to_string(index=False))
print(f"Süre: {time.time()-start:.2f}s")

plt.figure(figsize=(9, 5))
plt.barh(genre_pd["genre"][::-1], genre_pd["count"][::-1], color="#E74C3C")
plt.title("En Popüler Oyun Türleri (Genre, Top 10)")
plt.xlabel("İstek Sayısı")
plt.tight_layout()
plt.savefig("figures/top_genres.png", dpi=150)
plt.close()

spark.stop()
print("\nAnaliz tamamlandı! Grafikler figures/ klasörüne kaydedildi.")
