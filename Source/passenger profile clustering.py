#!/usr/bin/env python
# coding: utf-8

# ## Read

# In[1]:


# importing
import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import psycopg2
from datetime import datetime


# In[2]:


# connection to the database
# Create an engine instance


'''
alchemyEngine = create_engine("postgresql+psycopg2://postgres:Renfe2022@172.19.28.174:5433/SalesSystem", 
                                pool_recycle=-1,  
                              pool_pre_ping=True,  
                              pool_size = 1,
                              max_overflow=10
                             );
'''

connection_string = "user=postgres password=Renfe2022 host=172.19.28.174 port=5433 dbname=SalesSystem"


# In[3]:


# get the parameters

data_folder = "data"
cluster_folder = "cluster"

date_from_str = input("Enter the date to start from (YYYY-MM-DD): ")
try:
    date_from_obj = datetime.strptime(date_from_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit("Invalid date format")

date_to_str = input("Enter the dateto finish to (YYYY-MM-DD): ")
try:
    date_to_obj = datetime.strptime(date_to_str, "%Y-%m-%d").date()
except ValueError:
    sys.exit("Invalid date format")

try:
    num_clusters = int(input("Enter the number of clusters to create: "))
except ValueError:
    sys.exit("Not a valid number")


# In[4]:


# define datatypes to save memory
columns = [
    "document", "last_travel", "corridor_key", "nationality", "gender",
    "num_travels", "average_price", "price_stddev", "avg_advance_days", "avg_group_size",
    "travels_economy", "travels_business", "travels_workday", "travels_thu_sat", "travels_fri",
    "travels_mak_jed", "travels_mak_kaec", "travels_mak_mad", "travels_jed_kaec", "travels_jed_mad",
    "travels_kaec_mad", "travels_mak_kaia", "travels_jed_kaia", "travels_kaia_kaec", "travels_kaia_mad",
    "travels_jed_mak", "travels_kaec_mak", "travels_mad_mak", "travels_kaec_jed", "travels_mad_jed",
    "travels_mad_kaec", "travels_kaia_mak", "travels_kaia_jed", "travels_kaec_kaia",
    "travels_mad_kaia", "unique_routes", "travels_late_night", "travels_early_morning", "travels_morning",
    "travels_afternoon", "travels_evening", "travels_early_night", "purchases_app", "purchases_web",
    "purchases_tvm", "purchases_tom", "travels_ramadan", "travels_hajj", "travels_no_peak_season", "residency", "cluster"
]
#default datatype float 16
dtype_dict = {col:'Float32' for col in columns}

# string columns
string_columns = ["document", "residency", "last_travel", "corridor_key", "nationality", "gender", "cluster"]
for col in string_columns:
    dtype_dict[col] = 'str'

# int16 columns
int_columns = ["num_travels", "unique_routes"]
for col in int_columns:
    dtype_dict[col] = 'Int16'


# In[5]:


# creating customers table in database
print("Creating table customers in database (this may take long)")
try:
    with psycopg2.connect(connection_string) as conn:
        conn.cursor().execute("select from \"CRM\".create_customers_table(%s,%s)",(date_from_obj, date_to_obj))
        conn.commit()
except Exception as e:
    sys.exit(f"Error creating the customers table: {e}")
    


# In[6]:


# get the data
skip_reading = False

os.makedirs(data_folder, exist_ok=True)

try:
    print("Reading database")
    with psycopg2.connect(connection_string) as conn:
        # save the table to a csv
        if skip_reading:
            print("Skipping reading from database")
        else:
            with open(os.path.join(data_folder,'customers_full.csv'), "w", encoding="utf-8") as f:
                conn.cursor().copy_expert(r"""
                    COPY "CRM".customers TO STDOUT WITH (FORMAT CSV, DELIMITER E'\t', NULL '\N')
                """, f)
    
except Exception as e:
    sys.exit(f"Error reading the Customers table in the database. ({e})")


#read the csv to panda dataframe
print("Loading data")
df = pd.read_csv(os.path.join(data_folder,'customers_full.csv'), delimiter='\t', header=None, names=columns, dtype=dtype_dict, na_values=r'\N',  index_col=False,
            )


# In[7]:


# processing
# create new columns
print("Preparing Data")
try:
    df.drop(labels=["cluster"], inplace=True)
except:
    print("Cluster information not found in the data. Proceeding.")
    
df['is_female'] = (df['gender'] == 'Female').astype(int)
df['is_resident'] = (df['residency'] == 'Resident').astype(int)


# In[8]:


# explore 
df.shape


# In[9]:


df.head()


# ## Clustering per route

# In[10]:


# prepare data
# remove personal information
scalable_features = ['num_travels', 'average_price', 'price_stddev', 'avg_advance_days', 'avg_group_size']


# In[11]:


# clean routes
df.loc[df['corridor_key'] == 'R1- MAK-MAD','corridor_key'] = 'R1-MAK-MAD'
df.loc[df['corridor_key'] == 'R2- MAK-KAIA','corridor_key'] = 'R2-MAK-KAIA'
df.loc[df['corridor_key'] == 'R3- KAIA-MAD','corridor_key'] = 'R3-KAIA-MAD'
df.loc[df['corridor_key'] == 'R4- MAK-KAIA-MAD','corridor_key'] = 'R4-MAK-KAIA-MAD'
df.loc[df['corridor_key'] == 'R1 - MAK-MAD','corridor_key'] = 'R1-MAK-MAD'
df.loc[df['corridor_key'] == 'R2 - MAK-KAIA','corridor_key'] = 'R2-MAK-KAIA'
df.loc[df['corridor_key'] == 'R3 - KAIA-MAD','corridor_key'] = 'R3-KAIA-MAD'
df.loc[df['corridor_key'] == 'R4 - MAK-KAIA-MAD','corridor_key'] = 'R4-MAK-KAIA-MAD'


# In[12]:


# divide df
groups_route = df.groupby('corridor_key')

for name_route, df_route in groups_route:
    print(f"Clustering route {name_route}")
    df_route.drop(columns=['corridor_key'], axis=1, inplace=True)
    # scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(df_route[scalable_features]), index=df_route.index, columns=scalable_features)
    non_scalable = df_route.drop(columns=scalable_features)
    training_features = non_scalable.join(X_scaled)

    #remove extra columns before training
    training_features.drop(columns=["document", "residency", "last_travel", "gender", "nationality"], inplace=True)
    training_features.fillna(0, inplace=True)

    # apply k means
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, verbose=1)
    df_route['cluster'] = kmeans.fit_predict(training_features)

    # export data
    print(f"Saving data for {name_route}")
    with pd.ExcelWriter(os.path.join(cluster_folder,f"cluster_summary_{name_route}.xlsx")) as writer:
        for cluster_id, group in df_route.groupby('cluster'):
            # metrics
            desc = group.describe().T.round(2)
            desc.to_excel(writer, sheet_name=f"cluster_{cluster_id}_stats")
            
            # Top 10 nacionalities
            nat = (
                group['nationality']
                .value_counts(normalize=True)
                .head(10) * 100
            ).round(2).rename("percentage")
            
            nat_df = nat.reset_index().rename(columns={'index': 'nationality'})
            nat_df.to_excel(writer, sheet_name=f"cluster_{cluster_id}_nationalities", index=False)
            
    # save df
    df_route.insert(2, 'corridor_key', name_route)
    df_route.drop(labels=['is_female','is_resident'], axis=1).to_csv(f"Data\\customers_{name_route}.csv", index=False, header=True)


# In[13]:


# clean the customers table
print("Delete previous data in database")
with psycopg2.connect(connection_string) as conn:
    conn.cursor().execute('TRUNCATE "CRM".customers')
    conn.commit()
    


# In[14]:


# upload the new customers table
print("Loading new data to database")
for name_route in list(groups_route.groups.keys()):
    print(f"Uploading {name_route}")
    with open(os.path.join(data_folder,f"customers_{name_route}.csv"), "r", encoding="utf-8") as f:
        next(f)  # skip header
        with psycopg2.connect(connection_string) as conn:
            conn.cursor().copy_expert("COPY \"CRM\".customers FROM STDIN WITH CSV", f)
            conn.commit()


# In[15]:


print("Process finished.")

