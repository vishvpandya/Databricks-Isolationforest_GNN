# Databricks notebook source
# MAGIC %pip install kaggle
# MAGIC

# COMMAND ----------


import os

os.environ["KAGGLE_USERNAME"] = "vishvpandya"
os.environ["KAGGLE_KEY"] = "KGAT_ab75212b35b47827f73e49b0a7998bd9"

print("✅ Kaggle credentials configured")


# COMMAND ----------

## 3. Create schema (database)
spark.sql("""
CREATE SCHEMA IF NOT EXISTS workspace.aml_fraud
""")

# 4. Create volume for Bronze layer
spark.sql("""
CREATE VOLUME IF NOT EXISTS workspace.aml_fraud.bronze_volume
""")

print("✅ Schema & Bronze Volume created")


# COMMAND ----------

# 5. Download & unzip IBM AML dataset into Bronze volume
volume_path = "/Volumes/workspace/aml_fraud/bronze_volume"

os.system(f"""
cd {volume_path} &&
kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml &&
unzip -o ibm-transactions-for-anti-money-laundering-aml.zip &&
rm -f ibm-transactions-for-anti-money-laundering-aml.zip
""")

print("✅ IBM AML dataset downloaded & extracted")

# COMMAND ----------

bronze_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/workspace/aml_fraud/bronze_volume/HI-Small_Trans.csv")

display(bronze_df.limit(5))


# COMMAND ----------

# MAGIC %restart_python
# MAGIC

# COMMAND ----------

# Load raw CSVs into Spark (Bronze DataFrame)
# Load raw CSVs into Spark (Bronze DataFrame)
bronze_df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/workspace/aml_fraud/bronze_volume/*.csv")

# Rename columns with invalid characters for Delta compatibility
for old, new in [
    ("From Bank", "from_bank"),
    ("To Bank", "to_bank"),
    ("Amount Received", "amount_received"),
    ("Receiving Currency", "receiving_currency"),
    ("Amount Paid", "amount_paid"),
    ("Payment Currency", "payment_currency"),
    ("Payment Format", "payment_format"),
    ("Is Laundering", "is_laundering")
]:
    if old in bronze_df.columns:
        bronze_df = bronze_df.withColumnRenamed(old, new)

display(bronze_df.limit(5))


# COMMAND ----------

#  Write Bronze Delta table
bronze_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.aml_fraud.bronze_transactions")

print("✅ Bronze Delta table created: workspace.aml_fraud.bronze_transactions")

# COMMAND ----------

spark.table("workspace.aml_fraud.bronze_transactions") \
     .select("Timestamp") \
     .show(10, truncate=False)


# COMMAND ----------

