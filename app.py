import pandas as pd
import os
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui

RPM_COLUMN = "RPM(Curve nominal)"
IMPELLER_COLUMN = "Impeller(Curve)"
PRODUCT_NUMBER_COLUMN = "ProductNumber"
PRODUCT_NAME_COLUMN = "Product name"
RPM_PUMP_DATA_COLUMN = "RPM(Pump data)"
Q_COLUMN = "Q [m³/h]"
H_COLUMN = "H [m]"
P2_COLUMN = "P2 [kW]"
NPSH_COLUMN = "NPSH [m]"
RPM_REAL_COLUMN = "RPM(real)"

XML_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"
SELECTOR_VERSION = "8.0.0"
SKB_VERSION = "24.3.0.240819.2070"

# Define the path to the template file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
TEMPLATE_FILE = "SKB Blank Curve PSD - Power_Metric.xlsx"
TEMPLATE_PATH = os.path.join(BASE_DIR, TEMPLATE_FILE)

# Add page title and sidebar
app_ui = ui.page_sidebar(  
    ui.sidebar(
        ui.input_text("pump_family", "Pump Family"),  
        ui.input_file("input_file", "Choose CSV File", accept=[".csv"], multiple=False),
        ui.input_checkbox_group(
            "speed_correct",
            "Speed Correct Data?",
            choices=["Yes", "No"],
            selected=["Yes"],
        ),
        ui.input_action_button("create_psd_file", "Create PSD", class_="btn-primary"),
        ui.output_text("psd_creation"),
        ui.input_action_button("create_xml", "Create XML"),
        ui.output_text("xml_creation"),

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
            ui.input_select("speed_select", "Select Speed/Trim:", {}), 
            ui.output_plot("qh_preview"),  
            ui.output_plot("qp2_preview"),  
            ui.output_plot("qnpsh_preview"),  
        ),
        col_widths=(6, 6) # Two columns for the plots (12 max width)
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
        df[PRODUCT_NUMBER_COLUMN] = df[PRODUCT_NUMBER_COLUMN].ffill()
        df[PRODUCT_NUMBER_COLUMN] = pd.to_numeric(df[PRODUCT_NUMBER_COLUMN], errors='coerce').astype(int)     # Cast ProductNumber column to integer
        df[PRODUCT_NAME_COLUMN] = df[PRODUCT_NAME_COLUMN].ffill()
        df[RPM_COLUMN] = df[RPM_COLUMN].ffill()
        df[RPM_PUMP_DATA_COLUMN] = df[RPM_PUMP_DATA_COLUMN].ffill()

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
        grouped = df.groupby(PRODUCT_NUMBER_COLUMN)
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
        ui.update_select("model_select", choices=update_select_choices())  # Update the dropdown


    @reactive.effect
    def update_speed_dropdown():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            ui.update_select("speed_select", choices={})  # Clear the speed dropdown
            return

        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]

            # Determine whether to use 'RPM' or 'mm' terminology
            terminology, trims = determine_terminology_and_trims(selected_curve)
            
            # Update the dropdown with the appropriate terminology
            speeds = {str(trim): f"{trim} {terminology}" for trim in trims}
            speeds["All"] = f"All {terminology}"  # Add the "All" option
            ui.update_select("speed_select", choices=speeds, selected="All")  # Update the speed dropdown
        else:
            ui.update_select("speed_select", choices={})  # Clear the speed dropdown if no product is selected


    def determine_terminology_and_trims(selected_curve):
        """
        Determines whether the product is trimmable (Impeller) or VFD (RPM) and returns the terminology and trims.

        Args:
            selected_curve: The Curve object for the selected product.

        Returns:
            tuple: A tuple containing the terminology (str) and trims (list).
        """
        if len(selected_curve.df[RPM_COLUMN].unique()) > 1:  # If RPM is changing
            terminology = "RPM"
            trims = selected_curve.df[RPM_COLUMN].unique()
        elif IMPELLER_COLUMN in selected_curve.df.columns and len(selected_curve.df[IMPELLER_COLUMN].unique()) > 1:  # If Impeller is changing
            terminology = "mm"
            trims = selected_curve.df[IMPELLER_COLUMN].unique()
        else:
            print("Neither RPM nor Impeller is changing.")
            return None, None  # Return None if neither is changing

        return terminology, trims



    # Create class to hold each curve by product number/trim
    class Curve:
        # def __init__(self, pumpModel: str, dataframe):
        def __init__(self, partNumber: str, dataframe):
            # self.name = pumpModel
            self.name = partNumber
            self.df = dataframe
            self.model = dataframe.iloc[0][PRODUCT_NAME_COLUMN]
            self.pn = str(int(dataframe.iloc[0][PRODUCT_NUMBER_COLUMN]))
            self.frequency = dataframe.iloc[0]['Frequency']
            self.phase = dataframe.iloc[0]['Phase']
            self.trims = self.df[RPM_COLUMN].unique().tolist()
            self.trim_curves = {} 


        def __repr__(self) -> str:
            return f"Pump Model(name={self.name}, dataframe=\n{self.df})"
        

        def create_grouped_trim_curves(self):
            '''Group entire curve df by the trim/speed column'''
            grouped = self.df.groupby(RPM_COLUMN)
            
            # Check if there are multiple speeds
            if len(grouped) > 1:
                print(f"Multiple speeds found for part number {self.pn}: {list(grouped.groups.keys())}")

            for group_trim, trim_df in grouped:
                self.trim_curves[group_trim] = trim_df[[Q_COLUMN, H_COLUMN, P2_COLUMN, NPSH_COLUMN, RPM_PUMP_DATA_COLUMN, RPM_COLUMN, RPM_REAL_COLUMN]]


        def _apply_affinity_laws(self, row, n2, n1):
            # Affinity Laws function
            N1, N2 = n1, n2
            Q2 = row[Q_COLUMN] * (N2 / N1)
            H2 = row[H_COLUMN] * (N2 / N1)**2
            NPSH2 = row[NPSH_COLUMN] * (N2 / N1)**2
            P22 = row[P2_COLUMN] * (N2 / N1)**3
            rpm_pump_data = row[RPM_PUMP_DATA_COLUMN]
            rpm_curve_nom = row[RPM_COLUMN]
            rpm_curve_real = row[RPM_REAL_COLUMN]
            return pd.Series([Q2, H2, P22, NPSH2, rpm_pump_data, rpm_curve_nom, rpm_curve_real], 
                            index=[Q_COLUMN, H_COLUMN, P2_COLUMN, NPSH_COLUMN, RPM_PUMP_DATA_COLUMN, RPM_COLUMN, RPM_REAL_COLUMN])


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
            required_columns = [RPM_REAL_COLUMN, RPM_COLUMN, Q_COLUMN, H_COLUMN, P2_COLUMN, NPSH_COLUMN]
            for col in required_columns:
                if col not in self.df.columns:
                    raise KeyError(f"Required column '{col}' is missing in the DataFrame.")

            # Apply affinity laws row by row
            def apply_affinity_laws(row):
                n1 = row[RPM_REAL_COLUMN]
                n2 = row[RPM_COLUMN]
                if n1 == 0 or n2 == 0:  # Avoid division by zero
                    return row  # Return the original row if N1 or N2 is zero

                # print(f'Applying affinity laws for N1: {n1}, N2: {n2}')

                # Apply affinity laws
                row[Q_COLUMN] = row[Q_COLUMN] * (n2 / n1)
                row[H_COLUMN] = row[H_COLUMN] * (n2 / n1) ** 2
                row[P2_COLUMN] = row[P2_COLUMN] * (n2 / n1) ** 3
                row[NPSH_COLUMN] = row[NPSH_COLUMN] * (n2 / n1) ** 2
                row[RPM_REAL_COLUMN] = n2
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
    @reactive.event(input.model_select, input.speed_select)
    def qh_preview():        
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber and Speed from the dropdowns
        selected_product_number = input.model_select()
        selected_speed = input.speed_select()

        print(f"Selected ProductNumber: {selected_product_number}, Selected Speed: {selected_speed}")  # Debug print

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]

            # Determine whether to use 'RPM' or 'mm' terminology
            terminology, trims = determine_terminology_and_trims(selected_curve)
            if terminology is None or trims is None:
                ui.update_select("speed_select", choices={})  # Clear the speed dropdown
                return

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            if selected_speed == "All":
                # Plot all trims
                for trim in trims:
                    trim_df = selected_curve.df[selected_curve.df[f'{terminology}(Curve nominal)'] == trim]
                    ax.plot(
                        trim_df[Q_COLUMN],
                        trim_df[H_COLUMN],
                        marker='o',
                        linestyle='-',
                        label=f"{terminology}: {int(trim)} {terminology}"
                    )
            elif selected_speed:
                # Plot only the selected trim
                try:
                    trim = float(selected_speed)
                    trim_df = selected_curve.df[selected_curve.df[f'{terminology}(Curve nominal)'] == trim]
                    ax.plot(
                        trim_df[Q_COLUMN],
                        trim_df[H_COLUMN],
                        marker='o',
                        linestyle='-',
                        label=f"{terminology}: {int(trim)} {terminology}"
                    )
                except ValueError:
                    print(f"Invalid trim value: {selected_speed}")

            # Plot the data (example: Q vs H)
            ax.set_title(f"Q vs. H for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("Head (H) [m]")
            ax.grid(True)
            ax.legend()

            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid
        

    # Render the QP2-plot based on the selected model
    @render.plot
    @reactive.event(input.model_select, input.speed_select)
    def qp2_preview():
        
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber and Speed from the dropdowns
        selected_product_number = input.model_select()
        selected_speed = input.speed_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]

            # Use the helper function to determine terminology and trims
            terminology, trims = determine_terminology_and_trims(selected_curve)
            if terminology is None or trims is None:
                return None

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            if selected_speed == "All":
                # Plot all trims
                for speed, trim_df in reversed(list(selected_curve.trim_curves.items())):
                    ax.plot(
                        trim_df[Q_COLUMN],
                        trim_df[P2_COLUMN],
                        marker='o',
                        linestyle='-',
                        label=f"Speed: {int(speed)} RPM"
                    )
            elif selected_speed:
                # Plot only the selected trim
                try:
                    speed = int(float(selected_speed))
                    if speed in selected_curve.trim_curves:
                        trim_df = selected_curve.trim_curves[speed]
                        ax.plot(
                            trim_df[Q_COLUMN],
                            trim_df[P2_COLUMN],
                            marker='o',
                            linestyle='-',
                            label=f"Speed: {speed} RPM"
                        )
                except ValueError:
                    print(f"Invalid speed value: {selected_speed}")

            # Plot the data (example: Q vs H)
            ax.set_title(f"Q vs. P2 for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("Power (P2) [kW]")
            ax.grid(True)
            ax.legend()
            
            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid

    # Render the QNPSH-plot based on the selected model
    @render.plot
    @reactive.event(input.model_select, input.speed_select)
    def qnpsh_preview():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            return None
        
        # Get the selected ProductNumber and Speed from the dropdowns
        selected_product_number = input.model_select()
        selected_speed = input.speed_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]

            # Use the helper function to determine terminology and trims
            terminology, trims = determine_terminology_and_trims(selected_curve)
            if terminology is None or trims is None:
                return None

            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))

            if selected_speed == "All":
                # Plot all trims
                for speed, trim_df in reversed(list(selected_curve.trim_curves.items())):
                    ax.plot(
                        trim_df[Q_COLUMN],
                        trim_df[NPSH_COLUMN],
                        marker='o',
                        linestyle='-',
                        label=f"Speed: {int(speed)} RPM"
                    )
            elif selected_speed:
                # Plot only the selected trim
                try:
                    speed = int(float(selected_speed))
                    if speed in selected_curve.trim_curves:
                        trim_df = selected_curve.trim_curves[speed]
                        ax.plot(
                            trim_df[Q_COLUMN],
                            trim_df[NPSH_COLUMN],
                            marker='o',
                            linestyle='-',
                            label=f"Speed: {speed} RPM"
                        )
                except ValueError:
                    print(f"Invalid speed value: {selected_speed}")

            # Plot the data (example: Q vs H)
            ax.set_title(f"Q vs. NPSH for ProductNumber {input.model_select()}")
            ax.set_xlabel("Flow Rate (Q) [m³/h]")
            ax.set_ylabel("NPSH [m]")
            ax.grid(True)
            ax.legend()
            
            return fig # Return the figure to be rendered in the plot output
        else:
            print(f"Selected ProductNumber '{selected_product_number}' not found in group_objects.")
            return None  # Return nothing if the selection is invalid


    @reactive.calc
    def add_speed_data():
        pass

    @render.text
    @reactive.event(input.create_psd_file)
    def psd_creation():
        from openpyxl import load_workbook

        pump_family_name = input.pump_family()  # Get the pump family from the input
        output_psd_file = f"{pump_family_name} - Curve PSD.xlsx"

        # Create a working copy of the template
        output_file_path = create_working_copy(TEMPLATE_PATH, output_psd_file)
        wb = load_workbook(output_file_path)

        # Get group objects
        group_objects = get_group_objects()

        # Copy and rename worksheets
        copy_and_rename_worksheets(wb, group_objects)

        # Apply speed correction if enabled
        group_objects = apply_speed_correction(group_objects, "Yes" in input.speed_correct())

        # Fill PSD data
        for object_name in group_objects:
            curve = group_objects[object_name]
            curve_sheet = wb[curve.pn]
            fill_curve_data(curve_sheet, base_cell='D7', trims=curve.trims, trim_curves=curve.trim_curves)

        # Fill header data
        header_sheet = wb["Curve Header Data"]
        header_sheet['B7'] = input.pump_family()  # Fill Pump Family
        fill_header_data(header_sheet, base_cell='A1', group_objects=group_objects)

        # Save the workbook
        wb.save(output_file_path)

        return f"Created {output_psd_file}"


    @render.text
    @reactive.event(input.create_xml)
    def xml_creation():

        import xml.etree.ElementTree as ET

        group_objects = get_group_objects()

        # Apply speed correction if enabled
        group_objects = apply_speed_correction(group_objects, "Yes" in input.speed_correct())

        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]
            df_selected = selected_curve.df  # Get the DataFrame of the selected Curve object

        def add_namespace(elem_tag, xsi_namespace):
            """Adds namespace to root node."""
            XHTML_NAMESPACE = xsi_namespace
            XHTML = "{%s}" % XHTML_NAMESPACE
            NSMAP = {'xsi' : XHTML_NAMESPACE} # the default namespace (no prefix)
            return ET.Element(elem_tag, nsmap=NSMAP) # lxml only!
        
        def add_elem_from_dict(parent_elem, elem_dict={'':''}):
            """Takes elements inside elem_dict and adds as elements to parent_elem"""
            if elem_dict:
                for key, value in elem_dict.items():
                    elem = ET.SubElement(parent_elem, key)
                    elem.text = str(value)
            else:
                elem = ET.SubElement(parent_elem)

        def create_pump_curve_dict(model_obj) -> dict:
            """Returns dictionary of updated attributes to be converted to elements """

            # Calculate max_speed and min_speed from the model object's trims
            max_speed = int(max(model_obj.trims))
            min_speed = int(min(model_obj.trims))

            pumpCurve_dict = {
                'curveNumber': model_obj.pn,
                'speedRef': max_speed,
                'polesRef': model_obj.calc_poles,
                'hzRef': '60',
                'eyeCount': '1',
                'speedCurveNominal': max_speed,
                'speedCurveMin': min_speed,
                'speedCurveMax': max_speed,
                'diaImpInc': '1.0',
                # 'mcsfMaxRef': float(model_obj.get_mcsf('max')),
                # 'mcsfMinRef': float(model_obj.get_mcsf('min')),
                'speedVariableCurveMin': min_speed,
                'speedVariableCurveMax': max_speed,
                # 'optionalCurveType': 'Power',
                'optionalCurveType': 'Power',
                'flowStartHeadEnabled': 'true',
                'flowStartEtaEnabled': 'false',
                'flowStartPowerEnabled': 'false',
                'flowStartNPSHEnabled':'false',
                'flowStopNPSHEnabled':'false',
                'flowStartSubmergenceEnabled':'true',
                'extendNpshToMcsfMin':'false',
                'catalogTrimsSelectionMode':'0',
                'styleCurveBelowStart':'none',
                'flowExponentTrim':'1.0',
                'headExponentTrim':'2.0',
                'npshExponentTrim':'0.0',
                'etaExponentTrim':'0.0',
                'powerDriverFixed':'0.0', # Fixed Motor HP
                'quantityMotors':'1',
                'serviceFactorDriverFixed':'1.0',
                'serviceFactorDriverFixedUsed':'false',
                'flowExponentSpeed':'1.0',
                'headExponentSpeed':'2.0',
                'etaExponentSpeedReduced':'0.0',
                'etaExponentSpeedIncreased':'0.0',
                'npshExponentSpeedReduced':'2.0',
                'npshExponentSpeedIncreased':'2.0',
                'submergenceExponentSpeedReduced':'2.0',
                'submergenceExponentSpeedIncreased':'2.0',
                'hideEfficiencyInSelector':'false',
                'speedOfSoundRef':'331.6583',
                'speedOfSoundExpFlow':'1.0',
                'speedOfSoundExpHead':'2.0',
                'speedOfSoundExpEta':'0.0',
                'speedOfSoundExpEtaTotal':'0.0',
                'temperatureGasInletSkb':'20.0',
                'pressureGasInletSkb':'1.01325',
                'relativeHumidityGasSkb':'50.0',
                'diaRotatingElement':'0.0',
                'solveVariantDisplayStrategy':'2', # Rounding to nearest: 1, Round up: 2
                'flowStopPercentBEP':'0.0',
                'headMarginFixedDia':'3.048', # This is in meters
                'headMarginFixedDiaPercentage':'0.0',
                'submergenceVortexMin':'0.0',
                'submergenceStartupMin':'0.0',
                'thrustFactor':'0.0',
                'thrustFactorBalanced':'0.0',
                'displayBothDiameters':'false',
                # 'isoEfficiencyValues':'56:62:65:68',
                'moiFirstStage':'0.0',
                'moiAdditionalStage':'0.0',
                'moiPumpCoupling':'0.0',
                'flowMaxAllowedMinRef':'0.0',
                'flowMaxAllowedMaxRef':'0.0',
                'loadRadialRef':'0.0'
                }
            
            return pumpCurve_dict
        
        def get_impeller_trim(curve_number:str):
            # Filter the DataFrame based on a condition in 'column1'
            condition = df_selected[PRODUCT_NUMBER_COLUMN] == curve_number

            # result = df.loc[condition, 'Cylindrical\nimpeller']
            result = df_selected.loc[condition, RPM_COLUMN] 
            result = result.to_list()
            return(result[0])

        def create_impeller_dict(trim) -> dict:
            """Creates dict of attributes to add to each Impeller node. """
            
            # append_dict contains appropriate values for attributes/tags in updates_list
            # append_dict = copy_values_from_source_xml(updates_list, new_curve_number)
            # speed_or_trim = get_impeller_trim(curve_number)

            # If any of the below are important, add tag to updates_list
            impeller_dict = {
                'diameter': trim,
                'flowStartNPSH':'0.0,',
                'diameterHubSide':'0.0',
                'weight':'0.0',
                'surgeFlow':'0.0',
                'flowStartEta':'0.0',
                'flowStartHead':'0.0',
                'flowStartNPSH':'0.0',
                'flowStartNPSH0Percent':'0.0',
                'flowStartNPSHIncipient':'0.0',
                'flowStopNPSH':'0.0',
                'flowStopNPSH0Percent':'0.0',
                'flowStopNPSHIncipient':'0.0',
                'flowStartSubmergence':'0.0',
                'flowStartPower':'0.0',
                'powerShutoffFixedEnabled':'false',
                'powerShutoffFixed':'0.0',
                'bepFixedEnabled':'true',
                'solveVariantMin':'0.0',
                'solveVariantMax':'0.0',
                'minimumVolumetricEfficiency':'0.0',
                'minimumVolumetricEfficiencyRated':'0.0',
                'maximumDifferentialPressure':'0.0',
                'stopFlow':'0.0'
            }
            
            return(impeller_dict)
        
        def _add_curve_data_points(parent_elem, curve_type, df):
            """ Adds curve data points for flow/power, flow/head, flow/NPSH """

            # Opens relevant curve tab in PSD, and grabs flow, head, power, npsh columns
            # curve_data_df = pd.read_excel(psd_filepath,sheet_name=curve_number, header=7, skiprows=[8], usecols="D,E,L,S", nrows=50)
            # curve_data_df = pd.read_excel(psd_filepath,sheet_name=curve_number, names=['Flow','Head','Power','NPSH'], skiprows=[7,8], usecols="D,E,L,S", nrows=50)
            # curve_data_df = curve_data_df.dropna()


            # Iterate through curve data df and create dicts of each data point that will be added as nodes to output xml
            for _, row in df.iterrows():
            # for _, model_obj in pump_models.items():
                datapoint_elem = ET.SubElement(parent_elem, "DataPoint", disabled="false")
                
                if curve_type == 'Efficiency':
                    datapoint_dict = {
                        # 'x': metric_to_us(row['Flow'], "flow"),
                        # 'y': metric_to_us(row[curve_type], 'power'),
                        'x': row[Q_COLUMN],
                        'y': row['Eta1'],
                        'isOnCurve':'false',
                        'division':'false',
                        'useCubicSplines':'false',
                        'slopeEnabled':'false'
                    }
                
                elif curve_type == 'Power':
                    datapoint_dict = {
                        # 'x': metric_to_us(row['Flow'], "flow"),
                        # 'y': metric_to_us(row[curve_type], 'power'),
                        'x': row[Q_COLUMN],
                        'y': row[P2_COLUMN],
                        'isOnCurve':'false',
                        'division':'false',
                        'useCubicSplines':'false',
                        'slopeEnabled':'false'
                    }

                elif (curve_type == 'Head'):
                    datapoint_dict = {
                        # 'x': metric_to_us(row['Flow'], "flow"),
                        # 'y': metric_to_us(row[curve_type], 'distance'),
                        'x': row[Q_COLUMN],
                        'y': row[H_COLUMN],
                        'isOnCurve':'false',
                        'division':'false',
                        'useCubicSplines':'true',
                        'slopeEnabled':'false'
                    }

                elif (curve_type == 'NPSH'):
                    datapoint_dict = {
                        # 'x': metric_to_us(row['Flow'], "flow"),
                        # 'y': metric_to_us(row[curve_type], 'distance'),
                        'x': row[Q_COLUMN],
                        'y': row[NPSH_COLUMN],
                        'isOnCurve':'false',
                        'division':'false',
                        'useCubicSplines':'false',
                        'slopeEnabled':'false'
                    }

                else:
                    print(f'curve_type not allowed: {curve_type}')
                    
                add_elem_from_dict(datapoint_elem, datapoint_dict)

        def add_curve(parent_elem, curve_type:str, model_obj, trim):
            """ Creates <Curve> parent element, and adds specified curve to xml """
            curve_elem = ET.SubElement(parent_elem, 'Curve', type=curve_type)

            # Add Curve Data Points to Curve Element
            # _add_curve_data_points(curve_elem, curve_number, curve_type)
            curve_df = model_obj.trim_curves[trim]
            _add_curve_data_points(curve_elem, curve_type, curve_df)

        # Create Root Tag using Custom Fields
        root_ns = "http://www.w3.org/2001/XMLSchema-instance"
        curve_family_name = input.pump_family()
        root = add_namespace('SKBData', root_ns)
        qname = ET.QName(root_ns,"type")

        # Add <CurveFamily> node
        curveFamily_elem = ET.SubElement(root, "CurveFamily", attrib={
            'selectorVersion': "8.0.0",
            'skbVersion': "24.3.0.240819.2070"
        })

        # Use the helper function to determine terminology and trims
        terminology, _ = determine_terminology_and_trims(next(iter(group_objects.values())))

        # Set svDataType and interpDataType based on terminology
        if terminology == "RPM":
            sv_data_type = "speed"
            interp_data_type = "speed"
        elif terminology == "mm":
            sv_data_type = "impellerDiamter"
            interp_data_type = "impellerDiamter"
        else:
            sv_data_type = "unknown"
            interp_data_type = "unknown"

        header_dict = {
                'name': curve_family_name,
                'impellerType':'radialFlow',
                'svDataType':sv_data_type,
                'interpDataType':interp_data_type,
                'compressorConditionsInputTypeSkb':'speedOfSound',
                'flowTypeSkb':'volumetricFlow',
                'headTypeSkb':'head',
                'headMarginForFixedDiameter':'3.048',
                'submergenceMethod':'fixedValue',
                'errorFitMax':'1.5',
                'pumpType':'0',
                'interpQty':'3',
                'efficiencyPowerDataType':'pump',
                
        }

        add_elem_from_dict(curveFamily_elem, header_dict)

        # Here we need to iterate through each trim for each model for creating impeller tags
        for _, model_object in group_objects.items():
            # <pumpCurveCollection xsi:type="CentrifugalPumpCurveCollection"> This is the parent of each pump curve"
            pumpCurveCollection_elem = ET.SubElement(curveFamily_elem, 'pumpCurveCollection', {qname: "CentrifugalPumpCurveCollection"})

            # Creates pump curve elements
            curve_dict = create_pump_curve_dict(model_object)
            add_elem_from_dict(pumpCurveCollection_elem, curve_dict)

            # Here we are trying to current impeller trim/vfd speeds from original GPI curve export csv
            for trim in model_object.trims:
                # Add Impeller Elements
                impeller_elem = ET.SubElement(pumpCurveCollection_elem, 'Impeller')

                # impeller_dict = create_impeller_dict(row['Curve number'])
                impeller_dict = create_impeller_dict(trim)
                add_elem_from_dict(impeller_elem, impeller_dict)

                # Add Curve Elements
                add_curve(impeller_elem, "Head", model_object, trim)
                add_curve(impeller_elem, "Power", model_object, trim)
                # add_curve(impeller_elem, "Efficiency", row['Curve number'])
                add_curve(impeller_elem, "NPSH", model_object, trim)

        tree = ET.ElementTree(root)
        xml_name = f'{curve_family_name}.xml'
        tree.write(xml_name)

        return f"Created {xml_name}"


    def apply_speed_correction(group_objects, speed_correct_enabled):
        """
        Applies speed correction to all Curve objects in group_objects if enabled.

        Args:
            group_objects (dict): Dictionary of Curve objects.
            speed_correct_enabled (bool): Whether speed correction is enabled.
        """
        if speed_correct_enabled:
            print("Speed correction enabled. Applying speed correction...")
            for object_name, model_object in group_objects.items():
                model_object.speed_correct_data()
                print(f"Speed correction applied to {object_name}.")
        else:
            print("Speed correction not enabled. Proceeding without speed correction.")

        return group_objects  # Return the updated group_objects


    def extract_pump_family(product_name):
        """
        Extracts the pump family name from the product name.

        Args:
            product_name (str): The product name string (e.g., "NBE 015-070/6.93").

        Returns:
            str: The pump family name (e.g., "NBE").
        """
        if not isinstance(product_name, str) or not product_name:
            return None  # Return None if the input is invalid

        # Split the product name by spaces and return the first word
        return product_name.split()[0]


    @reactive.effect
    def update_pump_family():
        group_objects = get_group_objects()  # Get the group_objects dictionary
        if group_objects is None:
            print("No group objects available.")
            ui.update_text("pump_family", value="")  # Clear the pump_family field
            return

        # Get the selected ProductNumber from the dropdown
        selected_product_number = input.model_select()

        # Check if the selected ProductNumber exists in group_objects
        if selected_product_number in group_objects:
            selected_curve = group_objects[selected_product_number]

            # Extract the pump family name from the Product name column
            pump_family_name = extract_pump_family(selected_curve.model)  # Use the helper function
            ui.update_text("pump_family", value=pump_family_name)  # Update the pump_family field
        else:
            ui.update_text("pump_family", value="")  # Clear the pump_family field if no product is selected


    def calculate_speed_range(trims):
        """Calculate max and min speeds from trims."""
        return int(max(trims)), int(min(trims))


    def create_working_copy(template_path, output_filename):
        """
        Creates a working copy of the template file.

        Args:
            template_path (str): Path to the template file.
            output_filename (str): Name of the output file.

        Returns:
            str: Path to the working copy.
        """
        import shutil
        return shutil.copyfile(template_path, output_filename)


    def copy_and_rename_worksheets(workbook, group_objects):
        """
        Copies and renames worksheets for each curve.

        Args:
            workbook: OpenPyXL workbook object.
            group_objects (dict): Dictionary of Curve objects.
        """
        for object_name in group_objects:
            curve_pn = group_objects[object_name].pn
            workbook.copy_worksheet(workbook['NEW']).title = curve_pn


    def fill_curve_data(sheet, base_cell, trims, trim_curves):
        """
        Fills curve data into the worksheet.

        Args:
            sheet: OpenPyXL worksheet object.
            base_cell (str): Base cell for data entry.
            trims (list): List of trims/speeds.
            trim_curves (dict): Dictionary of trim DataFrames.
        """
        first_row_offset = 3
        for index, each_speed_trim in enumerate(trims):
            cell_name = "{}{}".format('A', 10 + index)
            sheet[cell_name].value = int(each_speed_trim)

            curve_data_df = trim_curves[each_speed_trim].reset_index()
            for key, value in curve_data_df.iterrows():
                sheet[base_cell].offset(first_row_offset + key, 0 + 21 * index).value = round(value[Q_COLUMN], 3)
                sheet[base_cell].offset(first_row_offset + key, 1 + 21 * index).value = round(value[H_COLUMN], 3)
                sheet[base_cell].offset(first_row_offset + key, 7 + 21 * index).value = round(value[Q_COLUMN], 3)
                sheet[base_cell].offset(first_row_offset + key, 8 + 21 * index).value = round(value[P2_COLUMN], 3)
                sheet[base_cell].offset(first_row_offset + key, 14 + 21 * index).value = round(value[Q_COLUMN], 3)
                sheet[base_cell].offset(first_row_offset + key, 15 + 21 * index).value = round(value[NPSH_COLUMN], 3)


    def fill_header_data(sheet, base_cell, group_objects):
        """
        Fills header data into the worksheet.

        Args:
            sheet: OpenPyXL worksheet object.
            base_cell (str): Base cell for data entry.
            group_objects (dict): Dictionary of Curve objects.
        """
        first_row_offset = 10
        for index, object_name in enumerate(group_objects):
            curve = group_objects[object_name]
            df = curve.df
            max_speed, min_speed = calculate_speed_range(curve.trims)

            sheet[base_cell].offset((first_row_offset + index), 0).value = curve.pn
            sheet[base_cell].offset((first_row_offset + index), 2).value = df.iloc[0][RPM_PUMP_DATA_COLUMN]
            sheet[base_cell].offset((first_row_offset + index), 3).value = curve.calc_poles
            sheet[base_cell].offset((first_row_offset + index), 4).value = curve.frequency
            sheet[base_cell].offset((first_row_offset + index), 6).value = df.iloc[0][RPM_COLUMN]
            sheet[base_cell].offset((first_row_offset + index), 7).value = min_speed
            sheet[base_cell].offset((first_row_offset + index), 8).value = max_speed
            sheet[base_cell].offset((first_row_offset + index), 9).value = '0'  # Diameter Increment
            sheet[base_cell].offset((first_row_offset + index), 15).value = 'Round up'  # Final diameter strategy
            sheet[base_cell].offset((first_row_offset + index), 21).value = min_speed


    output.df_preview = df_preview

app = App(app_ui, server)