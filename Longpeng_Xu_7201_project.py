'''
Project: Big Data Analytics on 2020 US Election
Author: Longpeng Xu (https://github.com/mattxu98)
Date: 20230526
'''
# Overall preprocessing

from pyspark.sql import SparkSession
import pyspark
spark = SparkSession \
    .builder \
    .appName('project') \
    .getOrCreate() 
# This returns a SparkSession object if already exists, and creates a new one if not exist.
spark

fb = spark.read.option("header",True).json('/data/ProjectDatasetFacebook/*').dropDuplicates()



# Trump preprocessing

trump = fb.filter(fb['page_name'] == 'Donald J. Trump')
trump.show()

from pyspark.sql.functions import to_date
trump = trump.withColumn("date_type", to_date("ad_delivery_start_time"))

trump.printSchema()

# Trump task 1

from pyspark.sql.functions import count, desc
trump_counts = trump.groupBy('date_type').agg(count('*').alias('count'))
trump_counts = trump_counts.orderBy('date_type')
trump_counts.show(1000)

from pyspark.sql.functions import col, max, expr, desc
trump = trump.withColumn("average_spend", (expr("CAST(spend.lower_bound AS INT)") + expr("CAST(spend.upper_bound AS INT)")) / 2)
trump_max_spend = trump.groupBy('id').agg(max('average_spend').alias('max_average_spend'))
trump_max_spend = trump_max_spend.orderBy(desc('max_average_spend')).limit(1000)
trump_max_spend.show(1000)

# Trump task 2

from pyspark.sql.functions import explode, avg, expr, concat_ws, year, month

trump_age = trump.withColumn('year', year('date_type')).withColumn('month', month('date_type'))
trump_age = trump_age.withColumn('year_month', concat_ws('-', 'year', 'month'))
trump_age = trump_age.select('year_month', explode('demographic_distribution').alias('demographics'))
trump_age = trump_age.select('year_month', 'demographics.age', expr("CAST(demographics.percentage AS FLOAT)").alias('percentage'))
trump_age = trump_age.filter(trump_age.age != '13-17')
trump_avg_pc = trump_age.groupBy('year_month', 'age').agg(avg('percentage').alias('average_percentage'))
trump_avg_pc = trump_avg_pc.orderBy('year_month','age').limit(1000)
trump_avg_pc.show(1000)

# Trump task 3

trump_words = trump.select('date_type', 'ad_creative_body')

from pyspark.sql.functions import split, lower
trump_words = trump_words.withColumn("ad_words", split(lower(trump_words["ad_creative_body"]), " "))

trump_words = trump_words.selectExpr("*", "explode(ad_words) as ad_words_exploded")

from pyspark.sql.functions import count, desc
trump_words = trump_words.groupBy('ad_words_exploded').agg(count('*').alias('count_'))

trump_words = trump_words.orderBy(desc('count_')).limit(50)

trump_words.show(50)



# Biden preprocessing

biden = fb.filter(fb['page_name'] == 'Joe Biden')
biden.show()

from pyspark.sql.functions import to_date
biden = biden.withColumn("date_type", to_date("ad_delivery_start_time"))

biden.printSchema()

# Biden task 1

from pyspark.sql.functions import count, desc
biden_counts = biden.groupBy('date_type').agg(count('*').alias('count'))
biden_counts = biden_counts.orderBy('date_type')
biden_counts.show(1000)

from pyspark.sql.functions import col, max, expr, desc
biden = biden.withColumn("average_spend", (expr("CAST(spend.lower_bound AS INT)") + expr("CAST(spend.upper_bound AS INT)")) / 2)
biden_max_spend = biden.groupBy('id').agg(max('average_spend').alias('max_average_spend'))
biden_max_spend = biden_max_spend.orderBy(desc('max_average_spend')).limit(1000)
biden_max_spend.show(1000)

# Biden task 2

from pyspark.sql.functions import explode, avg, expr, concat_ws, year, month

biden_age = biden.withColumn('year', year('date_type')).withColumn('month', month('date_type'))
biden_age = biden_age.withColumn('year_month', concat_ws('-', 'year', 'month'))
biden_age = biden_age.select('year_month', explode('demographic_distribution').alias('demographics'))
biden_age = biden_age.select('year_month', 'demographics.age', expr("CAST(demographics.percentage AS FLOAT)").alias('percentage'))
biden_age = biden_age.filter(biden_age.age != '13-17')
biden_avg_pc = biden_age.groupBy('year_month', 'age').agg(avg('percentage').alias('average_percentage'))
biden_avg_pc = biden_avg_pc.orderBy('year_month','age').limit(1000)
biden_avg_pc.show(1000)

# Biden task 3

biden_words = biden.select('date_type', 'ad_creative_body')

from pyspark.sql.functions import split, lower
biden_words = biden_words.withColumn("ad_words", split(lower(biden_words["ad_creative_body"]), " "))

biden_words = biden_words.selectExpr("*", "explode(ad_words) as ad_words_exploded")

from pyspark.sql.functions import count, desc
biden_words = biden_words.groupBy('ad_words_exploded').agg(count('*').alias('count_'))

biden_words = biden_words.orderBy(desc('count_')).limit(50)

biden_words.show(50)



# Plots
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Task 1a Plot

trump_counts_pandas = trump_counts.toPandas()
biden_counts_pandas = biden_counts.toPandas()
df1 = trump_counts_pandas
df2 = biden_counts_pandas

# Convert to datetime type
df1['date_type'] = pd.to_datetime(df1['date_type'])
df2['date_type'] = pd.to_datetime(df2['date_type'])

# Plot two series
plt.figure(figsize=(10,6), dpi=600, facecolor='white')

plt.plot(df1['date_type'], df1['count'], label='Series 1')
plt.plot(df2['date_type'], df2['count'], label='Series 2')

plt.xlabel('Year-Month')
plt.ylabel('Count')
plt.title('Task 1a: Frequencies of advertisements in the pre-election period')
plt.legend(['Donald J. Trump','Joe Biden'])

plt.show()

# Task 1b Plot

trump_max_spend_pandas = trump_max_spend.toPandas()
biden_max_spend_pandas = biden_max_spend.toPandas()
df3 = trump_max_spend_pandas
df4 = biden_max_spend_pandas

# Plot the data
fig = plt.figure(figsize=(12, 6), dpi=600, facecolor='white')

# Formatter function
def to_fixed(number, digits=5):
    return f"{number:.{digits}f}"
formatter = FuncFormatter(lambda y, _: to_fixed(y))

ax1 = fig.add_subplot(1, 2, 1)
ax1.hist(df3['max_average_spend'], density=True, bins=50, alpha=0.5)
ax1.set_title('Donald J. Trump')
ax1.set_xlabel('Average Spend in USD')
ax1.set_ylabel('Relative Frequency')
ax1.yaxis.set_major_formatter(formatter)

ax2 = fig.add_subplot(1, 2, 2)
ax2.hist(df4['max_average_spend'], density=True, bins=50, alpha=0.5)
ax2.set_title('Joe Biden')
ax2.set_xlabel('Average Spend in USD')
ax2.set_ylabel('Relative Frequency')
ax2.yaxis.set_major_formatter(formatter)

plt.suptitle('Task 1b: The distribution of 1000 greatest average spends')
plt.subplots_adjust(wspace=0.25, top=0.85)
plt.show()

# Task 2 Plot

trump_avg_pc_pandas = trump_avg_pc.toPandas()
biden_avg_pc_pandas = biden_avg_pc.toPandas()
df1 = trump_avg_pc_pandas
df2 = biden_avg_pc_pandas

# Pivot the data to have one column per age group
df1_pivot = df1.pivot(index='year_month', columns='age', values='average_percentage').fillna(0)
df2_pivot = df2.pivot(index='year_month', columns='age', values='average_percentage').fillna(0)

# Create a figure with 1*2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=600, facecolor='white')

# Plot df1
for column in df1_pivot.columns:
    axes[0].plot(df1_pivot.index, df1_pivot[column], label=column)
axes[0].set_xlabel('Year-Month')
axes[0].set_ylabel('Average Percentage')
axes[0].set_title('Donald J. Trump')
axes[0].legend(loc="lower left")

# Plot df2
for column in df2_pivot.columns:
    axes[1].plot(df2_pivot.index, df2_pivot[column], label=column)
axes[1].set_xlabel('Year-Month')
axes[1].set_ylabel('Average Percentage')
axes[1].set_title('Joe Biden')
axes[1].legend(loc="lower left")

# Rotate x-axis labels by 45 degrees for both subplots
axes[0].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='x', rotation=45)

# Supreme title
plt.suptitle('Task 2: Average percentage of each age group in pre-election period')
plt.show()