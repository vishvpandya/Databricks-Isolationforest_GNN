# Databricks notebook source
spark.sql("""
          CREATE VOLUME IF NOT EXISTS workspace.aml_fraud.silver_volume
""")
print("silver vloume ready")


# COMMAND ----------

bronze_df = spark.table("workspace.aml_fraud.bronze_transactions")
display(bronze_df.limit(5))


# COMMAND ----------

bronze_df.printSchema()


# COMMAND ----------

from pyspark.sql.functions import col

cols_to_drop = [c for c in bronze_df.columns if "unnamed" in c.lower() or c == "_c0"]

silver_df = bronze_df.drop(*cols_to_drop)

print("Dropped columns:", cols_to_drop)

# COMMAND ----------

from pyspark.sql.functions import to_timestamp, col

silver_df = silver_df \
    .withColumn("Timestamp", to_timestamp(col("Timestamp"), "yyyy/MM/dd HH:mm")) \
    .withColumn("amount_paid", col("amount_paid").cast("double")) \
    .withColumn("amount_received", col("amount_received").cast("double")) \
    .withColumn("is_laundering", col("is_laundering").cast("int"))

# COMMAND ----------

silver_df = silver_df.filter(
    (col("amount_paid") > 0) &
    (col("amount_received") > 0) &
    col("from_bank").isNotNull() &
    col("to_bank").isNotNull()
)

# COMMAND ----------

from pyspark.sql.functions import lower, trim

silver_df = silver_df \
    .withColumn("from_bank", trim(lower(col("from_bank")))) \
    .withColumn("to_bank", trim(lower(col("to_bank")))) \
    .withColumn("payment_currency", trim(lower(col("payment_currency")))) \
    .withColumn("receiving_currency", trim(lower(col("receiving_currency")))) \
    .withColumn("payment_format", trim(lower(col("payment_format"))))

# COMMAND ----------

from pyspark.sql.functions import abs

silver_df = silver_df.withColumn(
    "amount_diff",
    abs(col("amount_paid") - col("amount_received"))
)

# COMMAND ----------

silver_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.silver_transactions")

print("Silver Delta table created: workspace.aml_fraud.silver_transactions")

# COMMAND ----------

spark.table("workspace.aml_fraud.silver_transactions").groupBy("is_laundering").count().show()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.aml_fraud.silver_transactions;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE TABLE workspace.aml_fraud.silver_transactions;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   MIN(Timestamp), 
# MAGIC   MAX(Timestamp) 
# MAGIC FROM workspace.aml_fraud.silver_transactions;
# MAGIC

# COMMAND ----------

