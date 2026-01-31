# Databricks notebook source
silver_nodes = spark.table("workspace.aml_fraud.silver_node_features")
display(silver_nodes.limit(5))


# COMMAND ----------

feature_cols = [
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
]

if_df = silver_nodes.select(
    "node_id",
    *feature_cols
)

display(if_df.limit(5))


# COMMAND ----------

if_pd = if_df.toPandas()
if_pd = if_pd.fillna(0)


# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(if_pd[feature_cols])


# COMMAND ----------


from sklearn.ensemble import IsolationForest

iso = IsolationForest(
    n_estimators=200,
    contamination=0.02,   # assume ~2% risky banks
    random_state=42,
    n_jobs=-1
)

if_pd["anomaly_flag"] = iso.fit_predict(X_scaled)


# COMMAND ----------

if_pd["is_anomaly"] = (if_pd["anomaly_flag"] == -1).astype(int)


# COMMAND ----------

if_pd["anomaly_score"] = -iso.score_samples(X_scaled)


# COMMAND ----------

if_pd["is_anomaly"].value_counts()


# COMMAND ----------

if_result = if_pd[["node_id", "is_anomaly", "anomaly_score"]]

spark.createDataFrame(if_result).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.silver_node_anomalies")

print("✅ Isolation Forest results saved")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT is_anomaly, COUNT(*) 
# MAGIC FROM workspace.aml_fraud.silver_node_anomalies
# MAGIC GROUP BY is_anomaly;
# MAGIC

# COMMAND ----------

