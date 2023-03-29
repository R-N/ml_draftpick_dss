import plotly.express as px
from .util import aggregate
def plot_heroes(df, x="count", y="heroes"):
    heroes, sorting = aggregate(df, x, y)
    heroes = heroes[[x]].reset_index()

    fig = px.bar(
        heroes, 
        x, 
        y, 
        #color=df2.index, 
        orientation='h',
        hover_data=[x],
        category_orders=sorting.index.to_list(),
        #labels=labels
    )
    return fig
