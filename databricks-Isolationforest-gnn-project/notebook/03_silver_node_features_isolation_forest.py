# Databricks notebook source
silver_txn = spark.table("workspace.aml_fraud.silver_transactions")

display(silver_txn.limit(5))


# COMMAND ----------

from pyspark.sql.functions import count, sum, avg, countDistinct

outgoing = silver_txn.groupBy("from_bank").agg(
    count("*").alias("out_txn_count"),
    sum("amount_paid").alias("out_total_amount"),
    avg("amount_paid").alias("out_avg_amount"),
    countDistinct("to_bank").alias("out_unique_counterparties")
)


# COMMAND ----------

incoming = silver_txn.groupBy("to_bank").agg(
    count("*").alias("in_txn_count"),
    sum("amount_received").alias("in_total_amount"),
    avg("amount_received").alias("in_avg_amount"),
    countDistinct("from_bank").alias("in_unique_counterparties")
)


# COMMAND ----------

node_features = outgoing.join(
    incoming,
    outgoing.from_bank == incoming.to_bank,
    how="outer"
)


# COMMAND ----------

from pyspark.sql.functions import col, coalesce

node_features = node_features.withColumn(
    "node_id",
    coalesce(col("from_bank"), col("to_bank"))
)


# COMMAND ----------

node_features.select("node_id").distinct().show(10, truncate=False)


# COMMAND ----------

node_features = node_features.fillna(0)



# COMMAND ----------

node_features = node_features.withColumn(
    "in_out_amount_ratio",
    col("in_total_amount") / (col("out_total_amount") + 1)
).withColumn(
    "in_out_txn_ratio",
    col("in_txn_count") / (col("out_txn_count") + 1)
)


# COMMAND ----------

silver_nodes = node_features.select(
    "node_id",
    "out_txn_count",
    "out_total_amount",
    "out_avg_amount",
    "out_unique_counterparties",
    "in_txn_count",
    "in_total_amount",
    "in_avg_amount",
    "in_unique_counterparties",
    "in_out_amount_ratio",
    "in_out_txn_ratio"
)

display(silver_nodes.limit(5))


# COMMAND ----------

silver_nodes.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.silver_node_features")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.aml_fraud.silver_node_features;
# MAGIC

# COMMAND ----------

