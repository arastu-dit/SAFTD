import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import time
import learn
import twitter
import settings
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.plotly as py

#Init the model
MODEL,TOKENS = learn.init()

app = dash.Dash(__name__)
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

app.layout = html.Div(
    html.Div(children = [
        html.H1('Sentiment Analysis on Twitter using Machine Learning'),
        dcc.Interval(
            id='interval-component',
            interval=1000 * settings.REFRESH_RATE,
            n_intervals=0
        ),
        dcc.Input(id='tag', value='Liverpool', type='text'),
        html.Div([
            html.Div(id='tweets', className='six columns'),
            html.Div([
                dcc.Graph(id='perc'),
                dcc.Graph(id='count')
            ], className='six columns')
        ])
    ])
)

def generate_table(rows):
    return html.Table(
        # Header
        [html.Tr([html.Th('Sentiment'),html.Th('Tweet')])] +

        # Body
        [html.Tr([html.Td(row[0]),html.Td(row[1])]) for row in rows]
    )


@app.callback(
    Output(component_id='tweets', component_property='children'),
    [Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('tag', 'value')],
)
def update_output_div(n,tag):
    rows = [[learn.predict(MODEL, x,TOKENS), x.replace('\n','')] for x in twitter.get_last_tweets(tag) ]
    return [generate_table(rows)]

@app.callback(
    Output(component_id='count', component_property='figure'),
    [Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('tag', 'value')],
)
def update_output_div(n,tag):
    rows = [[learn.predict(MODEL, x,TOKENS), x.replace('\n','')] for x in twitter.get_last_tweets(tag) ]
    emos = [row[0] for row in rows]
    fig  = go.Figure(
            data=[
                go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[emos.count('positive'),emos.count('neutral'),emos.count('negative')]) ],
            layout=go.Layout(
                title='Points Accumulation',
                showlegend=False
            )
        )

    return fig

@app.callback(
    Output(component_id='perc', component_property='figure'),
    [Input('interval-component', 'n_intervals')],
    [dash.dependencies.State('tag', 'value')],
)
def update_output_div(n,tag):
    rows = [[learn.predict(MODEL, x,TOKENS), x.replace('\n','')] for x in twitter.get_last_tweets(tag) ]
    emos = [row[0] for row in rows]
    x=['Neutral', 'Negative','Positive']
    y=[emos.count('neutral'),emos.count('negative'),emos.count('positive')]
    fig  = go.Figure(
            data=[
                go.Pie(labels=x, values=y) ],
            layout=go.Layout(
                title='Points Accumulation',
                showlegend=False
            )
        )

    return fig



if __name__ == '__main__':
    app.run_server()
