# Databricks notebook source
silver_txn = spark.table("workspace.aml_fraud.silver_transactions")
silver_anomaly = spark.table("workspace.aml_fraud.silver_node_anomalies")

display(silver_anomaly.limit(5))


# COMMAND ----------

from pyspark.sql.functions import col
anomaly_nodes = silver_anomaly.filter(col("is_anomaly") == 1)

display(anomaly_nodes.limit(5))

# COMMAND ----------

gold_nodes = anomaly_nodes.select(
    col("node_id"),
    col("anomaly_score")
)

gold_nodes.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_nodes")

print("✅ Gold graph nodes created")


# COMMAND ----------

edges = silver_txn.join(
    anomaly_nodes,
    (silver_txn.from_bank == anomaly_nodes.node_id) |
    (silver_txn.to_bank == anomaly_nodes.node_id),
    how="inner"
)

# COMMAND ----------

# from pyspark.sql.functions import col

from pyspark.sql.functions import col

gold_edges = edges.select(
    col("from_bank").alias("src"),
    col("to_bank").alias("dst"),
    col("amount_paid").alias("amount"),
    col("Timestamp").alias("event_ts"),
    col("payment_format"),
    col("is_laundering").alias("label")   # 👈 FIX HERE
)

display(gold_edges.limit(5))

# COMMAND ----------

gold_edges.printSchema()

# COMMAND ----------

gold_edges.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges")

print("✅ Gold graph edges created")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.aml_fraud.gold_graph_nodes;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM workspace.aml_fraud.gold_graph_edges;

# COMMAND ----------

from pyspark.sql.functions import col

silver_txn = spark.table("workspace.aml_fraud.silver_transactions")
anomaly_nodes = spark.table("workspace.aml_fraud.silver_node_anomalies") \
    .filter(col("is_anomaly") == 1)

gold_edges_full = silver_txn.join(
    anomaly_nodes.select(col("node_id").alias("src_id")),
    silver_txn.from_bank == col("src_id"),
    "inner"
).join(
    anomaly_nodes.select(col("node_id").alias("dst_id")),
    silver_txn.to_bank == col("dst_id"),
    "inner"
).select(
    col("from_bank").alias("src"),
    col("to_bank").alias("dst"),
    col("amount_paid").alias("amount"),
    col("Timestamp").alias("event_ts"),
    col("payment_format"),
    col("is_laundering").alias("label")
)

gold_edges_full.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges_full")

print("✅ gold_graph_edges_full created (UNFILTERED)")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.aml_fraud.gold_graph_edges_full;
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, date_sub, max, lit

# get max timestamp from FULL graph
max_ts = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .select(max("event_ts").alias("max_ts")) \
    .collect()[0]["max_ts"]

print("Using max_ts:", max_ts)

# apply 150-day window
gold_edges_150d = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .filter(col("event_ts") >= date_sub(lit(max_ts), 150))

gold_edges_150d.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges")

print("✅ gold_graph_edges rebuilt for last 150 days")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.aml_fraud.gold_graph_edges;
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col

silver_txn = spark.table("workspace.aml_fraud.silver_transactions")
anomaly_nodes = spark.table("workspace.aml_fraud.silver_node_anomalies") \
    .filter(col("is_anomaly") == 1)

gold_edges_full = silver_txn.join(
    anomaly_nodes.select(col("node_id").alias("src_id")),
    silver_txn.from_bank == col("src_id"),
    "inner"
).join(
    anomaly_nodes.select(col("node_id").alias("dst_id")),
    silver_txn.to_bank == col("dst_id"),
    "inner"
).select(
    col("from_bank").alias("src"),
    col("to_bank").alias("dst"),
    col("amount_paid").alias("amount"),
    col("Timestamp").alias("event_ts"),
    col("payment_format"),
    col("is_laundering").alias("label")
)

gold_edges_full.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges_full")

print("✅ gold_graph_edges_full created (UNFILTERED)")

# COMMAND ----------

from pyspark.sql.functions import col, date_sub, max, lit

# get max timestamp from FULL graph
max_ts = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .select(max("event_ts").alias("max_ts")) \
    .collect()[0]["max_ts"]

print("Using max_ts:", max_ts)

# apply 150-day window
gold_edges_150d = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .filter(col("event_ts") >= date_sub(lit(max_ts), 100))

gold_edges_150d.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges")

print("✅ gold_graph_edges rebuilt for last 100 days")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.aml_fraud.gold_graph_edges;
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col

silver_txn = spark.table("workspace.aml_fraud.silver_transactions")
anomaly_nodes = spark.table("workspace.aml_fraud.silver_node_anomalies") \
    .filter(col("is_anomaly") == 1)

gold_edges_full = silver_txn.join(
    anomaly_nodes.select(col("node_id").alias("src_id")),
    silver_txn.from_bank == col("src_id"),
    "inner"
).join(
    anomaly_nodes.select(col("node_id").alias("dst_id")),
    silver_txn.to_bank == col("dst_id"),
    "inner"
).select(
    col("from_bank").alias("src"),
    col("to_bank").alias("dst"),
    col("amount_paid").alias("amount"),
    col("Timestamp").alias("event_ts"),
    col("payment_format"),
    col("is_laundering").alias("label")
)

gold_edges_full.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges_full")

print("✅ gold_graph_edges_full created (UNFILTERED)")

# COMMAND ----------

from pyspark.sql.functions import col, date_sub, max, lit

# get max timestamp from FULL graph
max_ts = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .select(max("event_ts").alias("max_ts")) \
    .collect()[0]["max_ts"]

print("Using max_ts:", max_ts)

# apply 150-day window
gold_edges_150d = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .filter(col("event_ts") >= date_sub(lit(max_ts), 100))

gold_edges_150d.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges")

print("✅ gold_graph_edges rebuilt for last 100 days")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.aml_fraud.gold_graph_edges;

# COMMAND ----------

from pyspark.sql.functions import col

silver_txn = spark.table("workspace.aml_fraud.silver_transactions")
anomaly_nodes = spark.table("workspace.aml_fraud.silver_node_anomalies") \
    .filter(col("is_anomaly") == 1)

gold_edges_full = silver_txn.join(
    anomaly_nodes.select(col("node_id").alias("src_id")),
    silver_txn.from_bank == col("src_id"),
    "inner"
).join(
    anomaly_nodes.select(col("node_id").alias("dst_id")),
    silver_txn.to_bank == col("dst_id"),
    "inner"
).select(
    col("from_bank").alias("src"),
    col("to_bank").alias("dst"),
    col("amount_paid").alias("amount"),
    col("Timestamp").alias("event_ts"),
    col("payment_format"),
    col("is_laundering").alias("label")
)

gold_edges_full.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges_full")

print("✅ gold_graph_edges_full created (UNFILTERED)")

# COMMAND ----------

from pyspark.sql.functions import col, date_sub, max, lit

# get max timestamp from FULL graph
max_ts = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .select(max("event_ts").alias("max_ts")) \
    .collect()[0]["max_ts"]

print("Using max_ts:", max_ts)

# apply 150-day window
gold_edges_150d = spark.table("workspace.aml_fraud.gold_graph_edges_full") \
    .filter(col("event_ts") >= date_sub(lit(max_ts), 50))

gold_edges_150d.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.gold_graph_edges")

print("✅ gold_graph_edges rebuilt for last 50 days")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) 
# MAGIC FROM workspace.aml_fraud.gold_graph_edges;

# COMMAND ----------

