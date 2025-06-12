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
        ui.input_action_button("preview_btn", "Preview", class_="btn-primary"),
        title="Inputs"
    ),
    # ui.input_checkbox_group(
    #     "stats",
    #     "Summary Stats",
    #     choices=["Row Count", "Column Count", "Column Names"],
    #     selected=["Row Count", "Column Count", "Column Names"],
    # ),
    # ui.output_data_frame("summary"),
    ui.output_data_frame("df_preview")
)  


# Define the server logic
def server(input, output, session):
    @reactive.calc
    def parsed_file():
        file = input.input_file() # This is a list of files, we only want the first one
        if file is None:
            return pd.DataFrame()
        
        # Access the file path from the first uploaded file
        file_path = file[0]["datapath"]

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, sep=";", index_col=False, skip_blank_lines=False)
        
        # Clean the DataFrame 
        df.replace(',','.', regex=True, inplace=True)
        df.dropna(how='all', inplace=True)

        # Forward fill productname and curve nominal columns for grouping 
        df['ProductNumber'] = df['ProductNumber'].ffill()
        df['RPM(Curve nominal)'] = df['RPM(Curve nominal)'].ffill()
        df['RPM(Pump data)'] = df['RPM(Pump data)'].ffill()

        return df

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
    

    # Create class to hold each curve by product number/trim
    class Curve:
        # def __init__(self, pumpModel: str, dataframe):
        def __init__(self, partNumber: str, dataframe):
            # self.name = pumpModel
            self.name = partNumber
            self.df = dataframe
            self.model = dataframe.iloc[0]['Product name']
            self.pn = str(int(dataframe.iloc[0]['ProductNumber']))
            self.frequency = dataframe.iloc[0]['Frequency']
            self.phase = dataframe.iloc[0]['Phase']
            self.trims = self.df['RPM(Curve nominal)'].unique().tolist()
            self.trim_curves = {} 


        def __repr__(self) -> str:
            return f"Pump Model(name={self.name}, dataframe=\n{self.df})"


        # def return_xy_points(self, curveType:str):
        #     '''
        #     Returns 2 lists: 1 - flow values, 2 - Head or Power or NPSH
            
        #     Parameters:
        #     curveType (str): The list of y-values. Available options are:
        #         - 'Head'
        #         - 'Power'
        #         - 'NPSH'
            
        #     Raises:
        #     ValueError: If the action is not one of the available options.
        #     '''
        #     curve_types = ['Head','Power','NPSH']

        #     if curveType not in curve_types:
        #         raise ValueError(f"Invalid Curve Type. Available options are: {', '.join(curve_types)}")
        #     else:
        #         x_values = self.df['Q'].tolist()
        #         y_values = self.df[curve_dict[curveType]].tolist()
        #         return x_values, y_values
        

        def create_grouped_trim_curves(self):
            '''Group entire curve df by the trim/speed column'''
            grouped = self.df.groupby('RPM(Curve nominal)')
            for group_trim, trim_df in grouped:
                self.trim_curves[group_trim] = trim_df[['Q [m³/h]','H [m]','P2 [kW]','NPSH [m]','RPM(Pump data)','RPM(Curve nominal)','RPM(real)']]


        def _apply_affinity_laws(self, row, n2, n1):
            # Affinity Laws function
            N1, N2 = n1, n2
            Q2 = row['Q [m³/h]'] * (N2 / N1)
            H2 = row['H [m]'] * (N2 / N1)**2
            NPSH2 = row['NPSH [m]'] * (N2 / N1)**2
            P22 = row['P2 [kW]'] * (N2 / N1)**3
            rpm_pump_data = row['RPM(Pump data)']
            rpm_curve_nom = row['RPM(Curve nominal)']
            rpm_curve_real = row['RPM(real)']
            return pd.Series([Q2, H2, P22, NPSH2, rpm_pump_data, rpm_curve_nom, rpm_curve_real], 
                            index=['Q [m³/h]', 'H [m]', 'P2 [kW]', 'NPSH [m]', 'RPM(Pump data)', 'RPM(Curve nominal)', 'RPM(real)'])


        def create_new_trim_df(self, n2):
            """
            Takes in speed n2 and applies affinity laws to max available existing trim to calculate new curve data

            Output: Adds a dataframe to self.trim_curves dictionary

            """
            # Finds max existing trim and uses that as n1
            n1 = self.max_trim

            # Check if n1 exists in trim_curves
            if n1 not in self.trim_curves:
                raise KeyError(f"Trim '{n1}' not found in trim_curves dictionary.")
            
            df1 = self.trim_curves[n1]

            # Apply the affinity laws to each row of df1 to create df2
            df2 = df1.apply(self._apply_affinity_laws, axis=1, args=(n2,n1))

            # Add new dataframe to dictionary trim_curves
            self.trim_curves.update({n2:df2}) 

            # Update Trims Property
            self.trims = list(self.trim_curves.keys())


        @property
        def max_trim(self):
            return max(self.trims)
        

        def speed_correct_data(self):
            """
            Speed corrects the data in self.df using affinity laws.
            N1 is taken from 'RPM(real)' and N2 is taken from 'RPM (Curve nominal)'.
            """
            # Ensure required columns exist
            required_columns = ['RPM(real)', 'RPM(Curve nominal)', 'Q [m³/h]', 'H [m]', 'P2 [kW]', 'NPSH [m]']
            for col in required_columns:
                if col not in self.df.columns:
                    raise KeyError(f"Required column '{col}' is missing in the DataFrame.")

            # Apply affinity laws row by row
            def apply_affinity_laws(row):
                n1 = row['RPM(real)']
                n2 = row['RPM(Curve nominal)']
                if n1 == 0 or n2 == 0:  # Avoid division by zero
                    return row  # Return the original row if N1 or N2 is zero

                # print(f'Applying affinity laws for N1: {n1}, N2: {n2}')

                # Apply affinity laws
                row['Q [m³/h]'] = row['Q [m³/h]'] * (n2 / n1)
                row['H [m]'] = row['H [m]'] * (n2 / n1) ** 2
                row['P2 [kW]'] = row['P2 [kW]'] * (n2 / n1) ** 3
                row['NPSH [m]'] = row['NPSH [m]'] * (n2 / n1) ** 2
                row['RPM(real)'] = n2
                return row

            # Update the DataFrame in place
            self.df = self.df.apply(apply_affinity_laws, axis=1)

            # Regenerate trim_curves after speed correction
            self.create_grouped_trim_curves()
        

        @property
        def calc_poles(self):
            """
            Calculate the number of poles based on the speed.
            """
            max_speed = max(self.trims)
            if max_speed <= 2200:
                return 4
            elif max_speed > 2200:
                return 2
            else:
                raise ValueError(f"Unexpected speed: {max_speed} for {self.pn}. Expected 2200 or 4000.")


    @render.data_frame
    @reactive.event(input.preview_btn)
    def df_preview():
        df = parsed_file()
        if df.empty:
            return pd.DataFrame()       
        
        # Make a model from each group
        grouped = df.groupby('ProductNumber')         # Group by the pump model column
        group_objects = {}   # Structure: group_objects (dict) where each key = partnumber, value = Curve Object 

        for partNumber, group_df in grouped:
            # print(f'partnumber: {partnumber}, group_df: {group_df}')
            group_object = Curve(partNumber=partNumber, dataframe=group_df)
            group_object.create_grouped_trim_curves()
            group_objects[(f'{partNumber}')] = group_object

        # Create a DataFrame for preview
        preview_data = {
            "Pump Model": [key for key in group_objects.keys()],
            "Number of Trims": [len(value.trims) for value in group_objects.values()],
        }
        preview_df = pd.DataFrame(preview_data)
        
        return preview_df

    output.df_preview = df_preview
    
app = App(app_ui, server)