# Before you begin



### Provide your credentials to the runtime

```
from google.colab import auth
auth.authenticate_user()
print('Authenticated')
```

## Optional: Enable data table display

Colab includes the ``google.colab.data_table`` package that can be used to display large pandas dataframes as an interactive data table.
It can be enabled with:

```
%load_ext google.colab.data_table
```

If you would prefer to return to the classic Pandas dataframe display, you can disable this by running:
```python
%unload_ext google.colab.data_table
```

# Use BigQuery via magics

The `google.cloud.bigquery` library also includes a magic command which runs a query and either displays the result or saves it to a variable as a `DataFrame`.

```
# Display query output immediately

%%bigquery --project yourprojectid
SELECT 
  COUNT(*) as total_rows
FROM `bigquery-public-data.samples.gsod`
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_rows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114420316</td>
    </tr>
  </tbody>
</table>
</div>



```
# Save output in a variable `df`

%%bigquery --project yourprojectid df
SELECT 
  COUNT(*) as total_rows
FROM `bigquery-public-data.samples.gsod`
```

```
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_rows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114420316</td>
    </tr>
  </tbody>
</table>
</div>



# Use BigQuery through google-cloud-bigquery

See [BigQuery documentation](https://cloud.google.com/bigquery/docs) and [library reference documentation](https://googlecloudplatform.github.io/google-cloud-python/latest/bigquery/usage.html).

The [GSOD sample table](https://bigquery.cloud.google.com/table/bigquery-public-data:samples.gsod) contains weather information collected by NOAA, such as precipitation amounts and wind speeds from late 1929 to early 2010.


### Declare the Cloud project ID which will be used throughout this notebook

```
project_id = '[your project ID]'
```

### Sample approximately 2000 random rows

```
from google.cloud import bigquery

client = bigquery.Client(project=project_id)

sample_count = 2000
row_count = client.query('''
  SELECT 
    COUNT(*) as total
  FROM `bigquery-public-data.samples.gsod`''').to_dataframe().total[0]

df = client.query('''
  SELECT
    *
  FROM
    `bigquery-public-data.samples.gsod`
  WHERE RAND() < %d/%d
''' % (sample_count, row_count)).to_dataframe()

print('Full dataset has %d rows' % row_count)
```

    Full dataset has 114420316 rows


### Describe the sampled data

```
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station_number</th>
      <th>wban_number</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>mean_temp</th>
      <th>num_mean_temp_samples</th>
      <th>mean_dew_point</th>
      <th>num_mean_dew_point_samples</th>
      <th>mean_sealevel_pressure</th>
      <th>num_mean_sealevel_pressure_samples</th>
      <th>mean_station_pressure</th>
      <th>num_mean_station_pressure_samples</th>
      <th>mean_visibility</th>
      <th>num_mean_visibility_samples</th>
      <th>mean_wind_speed</th>
      <th>num_mean_wind_speed_samples</th>
      <th>max_sustained_wind_speed</th>
      <th>max_gust_wind_speed</th>
      <th>max_temperature</th>
      <th>total_precipitation</th>
      <th>snow_depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1979.000000</td>
      <td>1883.000000</td>
      <td>1883.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>741.000000</td>
      <td>741.000000</td>
      <td>1776.000000</td>
      <td>1776.000000</td>
      <td>1950.000000</td>
      <td>1950.000000</td>
      <td>1922.000000</td>
      <td>241.000000</td>
      <td>1977.000000</td>
      <td>1793.000000</td>
      <td>91.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>505585.599293</td>
      <td>89647.441132</td>
      <td>1987.181405</td>
      <td>6.525518</td>
      <td>15.715513</td>
      <td>52.391865</td>
      <td>13.018696</td>
      <td>42.018694</td>
      <td>12.982475</td>
      <td>1015.278630</td>
      <td>11.506164</td>
      <td>967.396491</td>
      <td>11.979757</td>
      <td>11.529279</td>
      <td>12.737050</td>
      <td>6.903385</td>
      <td>12.969744</td>
      <td>12.297659</td>
      <td>24.829461</td>
      <td>44.059231</td>
      <td>0.065694</td>
      <td>8.929670</td>
    </tr>
    <tr>
      <th>std</th>
      <td>302491.187318</td>
      <td>27088.238467</td>
      <td>15.993488</td>
      <td>3.419259</td>
      <td>8.661563</td>
      <td>23.329842</td>
      <td>7.886977</td>
      <td>21.691902</td>
      <td>7.914005</td>
      <td>9.325395</td>
      <td>7.516349</td>
      <td>71.774000</td>
      <td>7.783445</td>
      <td>8.320365</td>
      <td>7.843476</td>
      <td>5.039771</td>
      <td>7.860417</td>
      <td>6.778960</td>
      <td>8.776778</td>
      <td>23.344521</td>
      <td>0.283467</td>
      <td>9.922707</td>
    </tr>
    <tr>
      <th>min</th>
      <td>10100.000000</td>
      <td>13.000000</td>
      <td>1933.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-91.400002</td>
      <td>4.000000</td>
      <td>-63.799999</td>
      <td>4.000000</td>
      <td>956.299988</td>
      <td>4.000000</td>
      <td>604.500000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>-96.900002</td>
      <td>0.000000</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>238255.000000</td>
      <td>99999.000000</td>
      <td>1978.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>39.299999</td>
      <td>7.000000</td>
      <td>30.349999</td>
      <td>7.000000</td>
      <td>1009.700012</td>
      <td>6.000000</td>
      <td>952.900024</td>
      <td>6.000000</td>
      <td>6.300000</td>
      <td>7.000000</td>
      <td>3.400000</td>
      <td>7.000000</td>
      <td>7.800000</td>
      <td>19.400000</td>
      <td>32.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>538980.000000</td>
      <td>99999.000000</td>
      <td>1990.000000</td>
      <td>7.000000</td>
      <td>16.000000</td>
      <td>55.000000</td>
      <td>8.000000</td>
      <td>44.000000</td>
      <td>8.000000</td>
      <td>1014.850006</td>
      <td>8.000000</td>
      <td>995.599976</td>
      <td>8.000000</td>
      <td>9.300000</td>
      <td>8.000000</td>
      <td>5.800000</td>
      <td>8.000000</td>
      <td>11.100000</td>
      <td>23.900000</td>
      <td>46.400002</td>
      <td>0.000000</td>
      <td>5.900000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>725273.500000</td>
      <td>99999.000000</td>
      <td>2000.000000</td>
      <td>10.000000</td>
      <td>23.000000</td>
      <td>69.800003</td>
      <td>23.000000</td>
      <td>56.700001</td>
      <td>23.000000</td>
      <td>1020.799988</td>
      <td>21.000000</td>
      <td>1010.299988</td>
      <td>22.000000</td>
      <td>13.825000</td>
      <td>23.000000</td>
      <td>9.100000</td>
      <td>23.000000</td>
      <td>15.900000</td>
      <td>28.900000</td>
      <td>60.799999</td>
      <td>0.010000</td>
      <td>11.600000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999999.000000</td>
      <td>99999.000000</td>
      <td>2010.000000</td>
      <td>12.000000</td>
      <td>31.000000</td>
      <td>105.099998</td>
      <td>24.000000</td>
      <td>80.500000</td>
      <td>24.000000</td>
      <td>1052.900024</td>
      <td>24.000000</td>
      <td>1037.099976</td>
      <td>24.000000</td>
      <td>99.400002</td>
      <td>24.000000</td>
      <td>57.299999</td>
      <td>24.000000</td>
      <td>68.000000</td>
      <td>66.000000</td>
      <td>98.599998</td>
      <td>5.910000</td>
      <td>51.200001</td>
    </tr>
  </tbody>
</table>
</div>



### View the first 10 rows

```
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station_number</th>
      <th>wban_number</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>mean_temp</th>
      <th>num_mean_temp_samples</th>
      <th>mean_dew_point</th>
      <th>num_mean_dew_point_samples</th>
      <th>mean_sealevel_pressure</th>
      <th>num_mean_sealevel_pressure_samples</th>
      <th>mean_station_pressure</th>
      <th>num_mean_station_pressure_samples</th>
      <th>mean_visibility</th>
      <th>num_mean_visibility_samples</th>
      <th>mean_wind_speed</th>
      <th>num_mean_wind_speed_samples</th>
      <th>max_sustained_wind_speed</th>
      <th>max_gust_wind_speed</th>
      <th>max_temperature</th>
      <th>max_temperature_explicit</th>
      <th>min_temperature</th>
      <th>min_temperature_explicit</th>
      <th>total_precipitation</th>
      <th>snow_depth</th>
      <th>fog</th>
      <th>rain</th>
      <th>snow</th>
      <th>hail</th>
      <th>thunder</th>
      <th>tornado</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>105780</td>
      <td>99999</td>
      <td>1968</td>
      <td>9</td>
      <td>13</td>
      <td>46.000000</td>
      <td>8</td>
      <td>44.200001</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.700000</td>
      <td>8.0</td>
      <td>15.3</td>
      <td>8.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>43.000000</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>0.02</td>
      <td>NaN</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25710</td>
      <td>99999</td>
      <td>1989</td>
      <td>5</td>
      <td>2</td>
      <td>51.299999</td>
      <td>23</td>
      <td>44.900002</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.600000</td>
      <td>23.0</td>
      <td>7.5</td>
      <td>23.0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>42.799999</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>475160</td>
      <td>99999</td>
      <td>2003</td>
      <td>4</td>
      <td>26</td>
      <td>45.200001</td>
      <td>16</td>
      <td>44.500000</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.500000</td>
      <td>16.0</td>
      <td>7.4</td>
      <td>16.0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>39.200001</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>476720</td>
      <td>99999</td>
      <td>1989</td>
      <td>12</td>
      <td>8</td>
      <td>51.599998</td>
      <td>4</td>
      <td>34.000000</td>
      <td>4.0</td>
      <td>1005.400024</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>41.900002</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>940040</td>
      <td>99999</td>
      <td>1991</td>
      <td>6</td>
      <td>9</td>
      <td>84.300003</td>
      <td>4</td>
      <td>75.199997</td>
      <td>4.0</td>
      <td>1009.900024</td>
      <td>4.0</td>
      <td>1009.099976</td>
      <td>4.0</td>
      <td>24.900000</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>9.9</td>
      <td>NaN</td>
      <td>79.199997</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>103250</td>
      <td>99999</td>
      <td>1976</td>
      <td>7</td>
      <td>23</td>
      <td>63.599998</td>
      <td>13</td>
      <td>48.599998</td>
      <td>13.0</td>
      <td>1022.099976</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.600000</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>48.200001</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>919280</td>
      <td>99999</td>
      <td>1981</td>
      <td>2</td>
      <td>21</td>
      <td>83.800003</td>
      <td>5</td>
      <td>75.900002</td>
      <td>5.0</td>
      <td>1007.900024</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.100000</td>
      <td>4.0</td>
      <td>8.5</td>
      <td>5.0</td>
      <td>8.9</td>
      <td>NaN</td>
      <td>77.000000</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>961710</td>
      <td>99999</td>
      <td>2004</td>
      <td>6</td>
      <td>23</td>
      <td>81.900002</td>
      <td>6</td>
      <td>74.900002</td>
      <td>6.0</td>
      <td>1010.500000</td>
      <td>6.0</td>
      <td>1008.000000</td>
      <td>6.0</td>
      <td>4.800000</td>
      <td>6.0</td>
      <td>1.5</td>
      <td>6.0</td>
      <td>5.1</td>
      <td>NaN</td>
      <td>74.800003</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>172400</td>
      <td>99999</td>
      <td>1990</td>
      <td>6</td>
      <td>6</td>
      <td>65.400002</td>
      <td>7</td>
      <td>43.900002</td>
      <td>7.0</td>
      <td>1018.200012</td>
      <td>7.0</td>
      <td>906.099976</td>
      <td>7.0</td>
      <td>18.600000</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>7.0</td>
      <td>4.1</td>
      <td>NaN</td>
      <td>45.000000</td>
      <td>False</td>
      <td>None</td>
      <td>None</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>38790</td>
      <td>99999</td>
      <td>1973</td>
      <td>4</td>
      <td>7</td>
      <td>44.799999</td>
      <td>8</td>
      <td>32.900002</td>
      <td>8.0</td>
      <td>1018.299988</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>17.200001</td>
      <td>8.0</td>
      <td>9.1</td>
      <td>8.0</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>35.599998</td>
      <td>True</td>
      <td>None</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```
# 10 highest total_precipitation samples
df.sort_values('total_precipitation', ascending=False).head(10)[['station_number', 'year', 'month', 'day', 'total_precipitation']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>station_number</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>total_precipitation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>644</th>
      <td>230220</td>
      <td>1964</td>
      <td>7</td>
      <td>15</td>
      <td>5.91</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>985430</td>
      <td>2008</td>
      <td>12</td>
      <td>8</td>
      <td>3.46</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>248260</td>
      <td>1961</td>
      <td>11</td>
      <td>1</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>1588</th>
      <td>257670</td>
      <td>1959</td>
      <td>8</td>
      <td>9</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>980</th>
      <td>299150</td>
      <td>1962</td>
      <td>3</td>
      <td>1</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>470250</td>
      <td>1965</td>
      <td>11</td>
      <td>25</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>1917</th>
      <td>288380</td>
      <td>1994</td>
      <td>8</td>
      <td>6</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>585190</td>
      <td>1995</td>
      <td>4</td>
      <td>14</td>
      <td>2.32</td>
    </tr>
    <tr>
      <th>250</th>
      <td>647000</td>
      <td>2005</td>
      <td>8</td>
      <td>19</td>
      <td>2.20</td>
    </tr>
    <tr>
      <th>1418</th>
      <td>964710</td>
      <td>1975</td>
      <td>9</td>
      <td>8</td>
      <td>1.97</td>
    </tr>
  </tbody>
</table>
</div>



# Use BigQuery through pandas-gbq

The `pandas-gbq` library is a community led project by the pandas community. It covers basic functionality, such as writing a DataFrame to BigQuery and running a query, but as a third-party library it may not handle all BigQuery features or use cases.

[Pandas GBQ Documentation](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_gbq.html)

```
import pandas as pd

sample_count = 2000
df = pd.io.gbq.read_gbq('''
  SELECT name, SUM(number) as count
  FROM `bigquery-public-data.usa_names.usa_1910_2013`
  WHERE state = 'TX'
  GROUP BY name
  ORDER BY count DESC
  LIMIT 100
''', project_id=project_id, dialect='standard')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>272793</td>
    </tr>
    <tr>
      <th>1</th>
      <td>John</td>
      <td>235139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Michael</td>
      <td>225320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Robert</td>
      <td>220399</td>
    </tr>
    <tr>
      <th>4</th>
      <td>David</td>
      <td>219028</td>
    </tr>
  </tbody>
</table>
</div>



# Syntax highlighting
`google.colab.syntax` can be used to add syntax highlighting to any Python string literals which are used in a query later.

```
from google.colab import syntax
query = syntax.sql('''
SELECT
  COUNT(*) as total_rows
FROM
  `bigquery-public-data.samples.gsod`
''')

pd.io.gbq.read_gbq(query, project_id=project_id, dialect='standard')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_rows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>114420316</td>
    </tr>
  </tbody>
</table>
</div>


