from darts.models import TransformerModel 
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.metrics import mape
import numpy as np
import altair as alt
import folium
import hyperparameter as config 
from utilities import *

def main():
    latlongdatapath = "DATA Path --- DISTRICT "
    covid_data = "DATA Path --- COVID"

    df = pd.read_csv(covid_data)
    latlong = pd.read_csv(latlongdatapath)

    no_of_district = 10
    fileName = None  # None for default.

    latlong['State'] = latlong['State'].str.upper()
    latlong['District'] = latlong['District'].str.upper()

    sum = 0

    # Create map
    map_sw = folium.Map(latitute=25., longitude=75.6872177)

    states_list = df['state_residence'].unique()

    for state in states_list:
        df1 = pd.read_csv(covid_data)
        df1 = df1.loc[df1['state_residence'] == str(state)]
        disticts = df1['district_residence'].unique()

        for distict in disticts:
            df = df1.loc[df1['district_residence'] == str(distict)]

            latlong_New = latlong.loc[latlong['State'] == str(state)]
            latlong_New = latlong_New.loc[latlong_New['District'] == str(distict)]

            if df.shape[0] < 50:
                scaled_pred_nhits = model.predict(n=config.predict_for)
                pred_nhits = train_scaler.inverse_transform(scaled_pred_nhits)
                actual = test.pd_dataframe().values.reshape(-1)
                predictions = pred_nhits.pd_dataframe().values.reshape(-1)
                actual_a = list(actual.reshape(-1))
                predict_list = list(int(x) for x in predictions)
                predict_list = [abs(x) for x in predict_list]

                actual_data = actual.reshape(-1)
                error = predictions - actual_data

                print(f'----------------------------------------------------------')
                print(f"My Actual of {distict} of {state} are {actual_a}")
                print(f"My Prediction of {distict} of {state} are {predictions}")
                print(f"My Error of {distict} of {state} are {error}")
                print(f'----------------------------------------------------------')

            else:
                # Data Processing
                df = df.groupby(['Date']).agg({'F_pos': 'sum', 'M_pos': 'sum', 'T_pos': 'sum', 'NIA_pos': 'sum', 'new_case': 'sum'}).reset_index()
                date = str(df.iloc[config.train_for + 1]['Date'])
                df_f = df['F_pos'].values
                df_m = df['M_pos'].values
                df_t = df['T_pos'].values
                df_nia = df['NIA_pos'].values
                df = df_f + df_m + df_t + df_nia

                series = TimeSeries.from_values(df)
                train, test = series[:config.train_for], series[config.train_for:config.train_for + config.predict_for]

                my_stopper = EarlyStopping(
                    monitor="train_loss",
                    patience=10,
                    min_delta=0.0001,
                    mode='min',)
                pl_trainer_kwargs = {"callbacks": [my_stopper]}

                model = TransformerModel(input_chunk_length=config.input_chunk_length
                                        , output_chunk_length=config.output_chunk_length
                                        , d_model=config.d_model
                                        , nhead=config.nhead
                                        , num_encoder_layers=config.num_encoder_layers
                                        , num_decoder_layers=config.num_decoder_layers
                                        , dim_feedforward=config.dim_feedforward
                                        , dropout=config.dropout
                                        , activation=config.activation,
                                        norm_type=None,
                                        custom_encoder=None,
                                        custom_decoder=None)

                train_scaler = Scaler()
                scaled_train = train_scaler.fit_transform(train)

                model.fit(scaled_train, epochs=config.epochs)

                scaled_pred_nhits = model.predict(n=config.predict_for)
                pred_nhits = train_scaler.inverse_transform(scaled_pred_nhits)
                actual = test.pd_dataframe().values.reshape(-1)
                predictions = pred_nhits.pd_dataframe().values.reshape(-1)
                actual_a = list(actual.reshape(-1))
                predict_list = list(int(x) for x in predictions)
                predict_list = [abs(x) for x in predict_list]

                actual_data = actual.reshape(-1)
                actual_data = remove_zeros(actual_data)
                print(f"MAPE for {distict} is {mape_(list(actual_data), predict_list)}")
                print(f'----------------------------------------------------------')

                # DATA FOR VISUALISATION
                data = pd.DataFrame({
                    'Time': np.arange(0, config.train_for + config.predict_for),
                    'Predictions': np.append(df[0:config.train_for], predict_list),
                    'Actual': np.append(df[0:config.train_for], actual_a)
                })

                # Initialize the result lists
                result_state_name = []
                result_distict_name = []
                result_mape_score = []
                
                # Reshape the data into a long format
                data = data.melt(id_vars=['Time'], value_vars=['Predictions', 'Actual'], var_name='Data', value_name='Number of covid cases')

                # Create the Altair chart
                chart = alt.Chart(data).mark_line().encode(
                    x='Time',
                    y=alt.Y('Number of covid cases:Q', title='Number of covid cases'),
                    color=alt.Color('Data:N', title='Data Type'),
                    tooltip=['Time', 'Number of covid cases']
                ).properties(
                    title=f"Predictions vs Actual results for {distict}",
                    width=600,
                    height=400
                )

                # Show the final chart
                chart.interactive()

                vega = folium.features.VegaLite(chart, width='100%', height='100%')

                latitude = latlong_New['Latitude'].values
                longitude = latlong_New['Longitude'].values
                result_state_name.append(state)
                result_distict_name.append(distict)
                result_mape_score.append(mape_(list(actual_data), predictions))

                if len(latitude) == 0 or len(longitude) == 0:
                    print(f"Location was not available for {distict}")
                else:
                    # create marker on the map, with optional popup text or Vincent visualization
                    sw_marker = folium.features.Marker([latitude, longitude], tooltip=str(distict))

                    # create popup
                    sw_popup = folium.Popup()

                    # add chart to popup
                    vega.add_to(sw_popup)

                    # add popup to marker
                    sw_popup.add_to(sw_marker)

                    # add marker to map
                    sw_marker.add_to(map_sw)
                    print(map_sw)
                    sum = sum + 1

                latitude = None
                longitude = None

    map_sw.save("save.html")

    stored_data = pd.DataFrame(data={
        'State_Name': result_state_name,
        'Distict': result_distict_name,
        'MAPE': result_mape_score})

if __name__ == "__main__":
    main()
