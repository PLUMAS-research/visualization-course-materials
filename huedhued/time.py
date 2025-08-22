import pandas as pd


def calcular_minutos(x):
    return pd.timedelta_range(x["inicio_actividad"], x["fin_actividad"], freq="min")


def contar_minutos(df):
    return df.assign(minutos=lambda x: x.apply(calcular_minutos, axis=1)).explode(
        "minutos"
    )


def calcular_matriz_de_tiempos(df):
    timeseries = pd.DataFrame(
        index=pd.timedelta_range(
            df["inicio_actividad"].min(), df["fin_actividad"].max(), freq="min"
        )
    )
    for idx, group in df.groupby("Proposito"):

        group_ts = (
            group.pipe(contar_minutos).groupby("minutos")["Peso"].sum().rename(idx)
        )
        timeseries = timeseries.join(group_ts, how="left")

    return timeseries.fillna(0)


def time_matrix(trips, max_trip_time=120):
    trips = trips[["Persona", "Proposito", "HoraIni", "TiempoViaje", "Peso"]]
    trips = trips[trips["TiempoViaje"] <= 120].copy()
    trips["inicio_actividad"] = (
        trips["HoraIni"] + pd.to_timedelta(trips["TiempoViaje"], unit="minutes")
    ).round("5min")

    activities = (
        trips.join(trips.add_prefix("post_").shift(-1))
        .pipe(lambda x: x[x["Persona"] == x["post_Persona"]])
        .drop(
            [
                "Persona",
                "post_Persona",
                "post_TiempoViaje",
                "post_Peso",
                "post_Proposito",
            ],
            axis=1,
        )
        .rename({"post_HoraIni": "fin_actividad"}, axis=1)
        .assign(fin_actividad=lambda x: x["fin_actividad"].round("5min"))
    )

    return activities.pipe(calcular_matriz_de_tiempos).fillna(0)
