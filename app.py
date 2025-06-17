import pandas as pd
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui

# Add page title and sidebar
app_ui = ui.page_sidebar(  
    ui.sidebar(
        ui.input_text("pump_family", "Pump Family", "SPE"),  
        ui.input_file("input_file", "Choose CSV File", accept=[".csv"], multiple=False),
        ui.input_checkbox_group(
            "speed_correct",
            "Speed Correct Data?",
            choices=["Yes", "No"],
            selected=["Yes"],
        ),
        ui.input_action_button("create_psd_file", "Create PSD", class_="btn-primary"),
        ui.output_text("psd_confirmation"),
        ui.input_action_button("create_xml", "Create XML"),
        title="Inputs",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Preview Pump Curve Data"),
            ui.input_select("model_select", "Select a Model to preview:", {}),    
            ui.output_data_frame("df_preview"),
        ), 
        ui.card(
            ui.card_header("Curve Data Preview"),
            ui.output_plot("qh_preview"),  
            ui.output_plot("qp2_preview"),  
            ui.output_plot("qnpsh_preview"),  
        ),
        col_widths=(8, 4)
    ),
)   

# Define the server logic
def server(input, output, session):
    @reactive.calc
    def parsed_file():
        file = input.input_file() # This is a list of files, we only want the first one
        if file is None:
            print("No input file selected.")
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
        df['ProductNumber'] = pd.to_numeric(df['ProductNumber'], errors='coerce').astype(int)     # Cast ProductNumber column to integer
        df['Product name'] = df['Product name'].ffill()
        df['RPM(Curve nominal)'] = df['RPM(Curve nominal)'].ffill()
        df['RPM(Pump data)'] = df['RPM(Pump data)'].ffill()

        return df
    

    @reactive.calc
    def get_group_objects():
        """
        Get the selected curve based on the input from the dropdown.
        This function is used to retrieve the Curve object for the selected ProductNumber.
        """
        df = parsed_file()
        if df.empty:
            print("No data available in the DataFrame.")
            return None
        
        # Group by 'ProductNumber' and create group_objects
        grouped = df.groupby('ProductNumber')
        group_objects = {}   # group_objects (dict) where each key = partnumber, value = Curve Object
        for partNumber, group_df in grouped:
            group_object = Curve(partNumber=partNumber, dataframe=group_df)
            group_object.create_grouped_trim_curves()
            group_objects[(f'{partNumber}')] = group_object

        return group_objects  # Return the dictionary of Curve objects
    

    @reactive.calc
    def update_select_choices():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return {}

        # Return the choices for the dropdown based on group_objects keys
        return {key: f"Partnumber {key}" for key in group_objects.keys()}


    # Bind the dropdown choices to the output of update_select_choices
    @reactive.effect
    def update_dropdown():
        # choices = update_select_choices()  # Get the updated choices
        ui.update_select("model_select", choices=update_select_choices())  # Update the dropdown


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
    @reactive.event(input.model_select)
    def df_preview():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:   
            print("No group objects available.")
            return {}
        
        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]
            return selected_curve.df  # Return the DataFrame of the selected Curve object
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return pd.DataFrame()  # Return an empty DataFrame if the selection is invalid

    # Render the QH-plot based on the selected model
    @render.plot
    @reactive.event(input.model_select)
    def qh_preview():        
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]
            df_selected = selected_curve.df  # Get the DataFrame of the selected Curve object

            # Plot the data (example: Q vs H)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df_selected['Q [m³/h]'], df_selected['H [m]'], marker='o', linestyle='-', color='b')
            ax.set_title(f"Q vs. H for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("Head (H) [m]")
            ax.grid(True)
            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid
        

    # Render the QP2-plot based on the selected model
    @render.plot
    @reactive.event(input.model_select)
    def qp2_preview():
        
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]
            df_selected = selected_curve.df  # Get the DataFrame of the selected Curve object

            # Plot the data (example: Q vs H)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df_selected['Q [m³/h]'], df_selected['P2 [kW]'], marker='o', linestyle='-', color='b')
            ax.set_title(f"Q vs. P2 for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("Power (P2) [kW]")
            ax.grid(True)
            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid


    # Render the QNPSH-plot based on the selected model
    @render.plot
    @reactive.event(input.model_select)
    def qnpsh_preview():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]
            df_selected = selected_curve.df  # Get the DataFrame of the selected Curve object

            # Plot the data (example: Q vs H)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(df_selected['Q [m³/h]'], df_selected['NPSH [m]'], marker='o', linestyle='-', color='b')
            ax.set_title(f"Q vs. NPSH for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("NPSH [m]")
            ax.grid(True)
            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid

    @reactive.calc
    def speed_correct():
        pass

    @render.text
    @reactive.event(input.create_psd_file)
    def psd_confirmation():
        return "PSD file created successfully!"

    output.df_preview = df_preview

app = App(app_ui, server)