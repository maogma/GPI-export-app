import pandas as pd
from shiny import App, reactive, render, ui

# Add page title and sidebar
app_ui = ui.page_sidebar(  
    ui.sidebar(
        ui.input_text("text", "Pump Family", "i.e. SPE, NB, Alpha..."),  
        ui.input_file("input_file", "Choose CSV File", accept=[".csv"], multiple=False),
        ui.input_checkbox_group(
            "file_outputs",
            "Desired File Outputs",
            choices=["Curve PSD","XML File"],
            selected=["Curve PSD"],
        ),
        ui.input_action_button("submit", "Create File(s)", class_="btn-primary"),
        title="Inputs"
    ),
    ui.input_checkbox_group(
        "stats",
        "Summary Stats",
        choices=["Row Count", "Column Count", "Column Names"],
        selected=["Row Count", "Column Count", "Column Names"],
    ),
    ui.output_data_frame("summary"),
    # ui.input_text("text", "Pump Family", "i.e. SPE, NB, Alpha..."),  
    # ui.output_text_verbatim("value")
)  


def server(input, output, session):
    @reactive.calc
    def parsed_file():
        file = input.input_file()
        if file is None:
            return pd.DataFrame()
        return pd.read_csv(file[0]["datapath"])

    @render.data_frame
    def summary():
        df = parsed_file()

        if df.empty:
            return pd.DataFrame()

        # Get the row count, column count, and column names of the DataFrame
        row_count = df.shape[0]
        column_count = df.shape[1]
        names = df.columns.tolist()
        column_names = ", ".join(str(name) for name in names)

        # Create a new DataFrame to display the information
        info_df = pd.DataFrame({
            "Row Count": [row_count],
            "Column Count": [column_count],
            "Column Names": [column_names],
        })

        # input.stats() is a list of strings; subset the columns based on the selected
        # checkboxes
        return info_df.loc[:, input.stats()]
    
    @render.text
    def value():
        return input.text()
    
    @render.text()
    @reactive.event(input.action_button)
    def counter():
        return f"{input.action_button()}"

app = App(app_ui, server)