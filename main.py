################################################### Importing libraries ################################################
# Core Packages
import io, os, shutil, re, time, pickle, pathlib, glob, base64, xlsxwriter
from io import BytesIO
from os import path
import missingno as msno
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import missingno as msno
import matplotlib.pyplot as plt
#warnings.filterwarnings("ignore")
## Disable Warning
st.set_option('deprecation.showfileUploaderEncoding', False)

################################################### Defining Static Data ###############################################

# static_store = get_static_store()
# session = SessionState.get(run_id=0)
st.sidebar.image('https://i.flockusercontent2.com/q884s08qs8s9s4lb?r=1157408321', use_column_width=False)
st.sidebar.markdown("<marquee >By **Ashish Gopal**</marquee>", unsafe_allow_html=True)

############################################################################################
if st.sidebar.checkbox("Want a Background?"):
    st.markdown(
        f"<style> "
        f"body{{ background-image: url(https://media0.giphy.com/media/Cv7wrQjYcd6hO/giphy.webp?cid=ecf05e47ikkphe1fxlvbw6dnacalpyo15imqrml92na8v738&rid=giphy.webp);}}"            
        f"</style>",
        unsafe_allow_html=True)

# user_color = st.sidebar.beta_color_picker("Pick any color to set the Background Color for Heading")
user_color='#000000'
html_temp = """
<div style="background-color:{};padding:12px">
<h1 style="color:white;text-align:center;">Complete Data Profiling Application</h1>
</div>
""".format(user_color)
st.markdown(html_temp, unsafe_allow_html=True)

## To hide the Streamlit menu
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
"""

if st.sidebar.checkbox('Brand Menu Hide'):
    st.markdown(hide_st_style, unsafe_allow_html=True)
################################################### Defining Static Paths ##############################################

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'

## We create a downloads directory within the streamlit static asset directory and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
if not DOWNLOADS_PATH.is_dir():
    DOWNLOADS_PATH.mkdir()

folder_names = [name for name in ["Raw Data" , "Modified Data", "Models"]]
if st.sidebar.checkbox('Click to Clear out all the data'):
    for folder_name in folder_names:
        ## Make the directory empty to fetch only required files
        if os.path.exists(os.path.join(DOWNLOADS_PATH, folder_name)):
            shutil.rmtree(os.path.join(DOWNLOADS_PATH, folder_name), ignore_errors=True)

for folder_name in folder_names:
    ## Make the directory empty to fetch only required files
    if not os.path.exists(os.path.join(DOWNLOADS_PATH,folder_name)):
        os.makedirs(os.path.join(DOWNLOADS_PATH,folder_name))

datafile_path       = os.path.join(DOWNLOADS_PATH, "Raw Data", "data.csv")
modifiedfile_path   = os.path.join(DOWNLOADS_PATH, "Modified Data", "data.csv")

## Path to store the Model
model_path          = os.path.join(DOWNLOADS_PATH, "Models")

model_folder_names = [name for name in ["Regression" , "Classification", "Others"]]
for folder_name in model_folder_names:
    if not os.path.exists(os.path.join(model_path,folder_name)):
        os.makedirs(os.path.join(model_path,folder_name))

# model_saved_path = str(DOWNLOADS_PATH / "model.sav")

################################################### Defining Functions #################################################

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index = False, sheet_name='Sheet1',float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Predictions.xlsx">Download Excel file</a>' # decode b'abc' => abc

################################################################################

# @st.cache(suppress_st_warning=True)
def load_data():
    data_df = pd.DataFrame()
    if path.exists(datafile_path):
        data_df = pd.read_csv(datafile_path)
        if st.checkbox("Click to view data"):
            st.write(data_df)
    return data_df

################################################################################

def load_modified_data():
    data_df = pd.DataFrame()
    if (not os.path.exists(datafile_path)) & (os.path.exists(modifiedfile_path)):
        os.remove(modifiedfile_path)
    if path.exists(modifiedfile_path):
        data_df = pd.read_csv(modifiedfile_path)
        if st.checkbox("Click to view Modified data"):
            st.write(data_df)
    return data_df

################################################################################

# @st.cache(suppress_st_warning=True)
def profile_report(df):
    from pandas_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    report = ProfileReport(df, minimal=True)
    st_profile_report(report)

################################################################################

def summary_data(df):
    st.write('---------------------------------------------------')
    if not df.empty:
        ## Details of Data
        st.write('#######################################')
        if st.checkbox("Top 20 rows"):
            st.dataframe(df.head(20))
        if st.checkbox("Bottom 20 rows"):
            st.dataframe(df.tail(20))
        if st.checkbox("Show Shape"):
            st.write(df.shape)
        if st.checkbox("Show Columns"):
            all_columns = df.columns = df.columns.to_list()
            st.write(all_columns)
        if st.checkbox("Show Summary"):
            st.write(df.describe(include='all'))
        if st.checkbox("Show Column Datatypes"):
            st.write(df.dtypes)
        if st.checkbox("Data Profiling (Takes few minutes to execute)"):
            profile_report(df)
        st.write('#######################################')
    else:
        st.markdown('**No Data Available to show!**.')
    st.write('---------------------------------------------------')

################################################################################

def preprocess_data(data_df):
    st.write('---------------------------------------------------')
    if not data_df.empty:
        all_columns = data_df.columns.to_list()
        if st.checkbox("Data Preprocess (Keep checked in to add steps)"):
            ## Receive a function to be called for Preprocessing
            df = data_df.copy()
            txt = st.text_area(
                "Provide lines of code in the given format to preprocess the data, otherwise leave it as commented",
                "## Consider the dataframe to be stored in 'df' variable\n" + \
                "## for e.g.\n" + \
                "## df['col_1'] = df['col_1'].astype('str')")
            if st.button("Finally, Click here to update the file"):
                exec(txt)
                if os.path.exists(modifiedfile_path):
                    os.remove(modifiedfile_path)
                df.to_csv(modifiedfile_path, index=False)
                st.success("New file created successfully under: {}".format(modifiedfile_path))
            if st.checkbox("Click to view Modified file"):
                if os.path.exists(modifiedfile_path):
                    st.write(pd.read_csv(modifiedfile_path))
                else:
                    st.markdown('**No Data Available to show!**.')
    else:
        st.markdown('**No Data Available to show!**.')
    st.write('---------------------------------------------------')

################################################################################

def visualization_data(df):
    df = df.copy()
    txt = "#"
    st.write('---------------------------------------------------')
    if not df.empty:
        all_columns = df.columns.to_list()
        if st.checkbox("Plot to show NULL values in Data"):
            selected_columns = st.multiselect("Select Columns ", all_columns)
            if len(selected_columns) > 0:
                new_df = df[selected_columns].copy()
                msno.matrix(new_df)
                st.pyplot()
        if st.checkbox("Visualize a particular column"):
            st.subheader("Data Preprocessing step")
            ## Receive a function to be called for Preprocessing
            txt = st.text_area(
                "Provide lines of code in the given format to preprocess the data",
                "## Consider the dataframe to be stored in 'df' variable\n" + \
                "## for e.g.\n" + \
                "## df['col_1'] = df['col_1'].astype('str')")
            plot_col_x = st.selectbox("Select Columns for X axis", all_columns)
            plot_col_y = st.multiselect("Select Columns for Y axis", all_columns)
            if (plot_col_x is not None) | (plot_col_y is not None):
                exec(txt)
                # Raw data plot of Mositure variable
                st.write("{} with respect to {}".format(plot_col_x, plot_col_y))
                plt.figure(figsize=(18, 4))
                plt.xlabel(plot_col_x)
                # plt.xticks(df[plot_col_x])
                columns_selected = [plot_col_x] + plot_col_y
                st.write(df[columns_selected])
                for i in range(len(plot_col_y)):
                    plt.plot(df[plot_col_y[i]], label=plot_col_y)
                plt.legend()
                # plt.ylabel(plot_col_y)
                st.pyplot()
            else:
                st.write("Please Select Columns")
    else:
        st.markdown('**No Data Available to show!**.')
        st.write("Did you run the Data Preprocess Step? if not, first run and then try again.")
    st.write('---------------------------------------------------')

################################################### Defining Main() #################################################

def main():
    activities_outer = ["Data Analysis", "About"]
    choice_1_outer = st.sidebar.radio("Choose your Step:", activities_outer)

    data = pd.DataFrame()
    dataset_flag = 0
    ############################################################################################
    if choice_1_outer == "Data Analysis":
        file_types = ["csv"]

        activities_1 = ["1. Data Import", "2. Data Summary", "3. Data Preprocess", "4. Data Visualization"]
        choice_1 = st.selectbox("Select Activities", activities_1)

        ############################################################################################
        if choice_1 == "1. Data Import":
            data = None
            show_file = st.empty()
            ############################################################################################
            if st.checkbox("Click to Upload data"):
                data = st.file_uploader("Upload Dataset : ",type=file_types)

                if st.checkbox('Click to load predefined dataset'):
                    dataset_path = os.path.join(os.path.abspath(__file__+ "/../"), "dataset")
                    files_available = [f for f in glob.glob(os.path.join(dataset_path,"*.csv"))]
                    files_name = [f.replace(dataset_path,"").replace("\\","") for f in glob.glob(os.path.join(dataset_path,"*.csv"))]
                    dataset_dropdown = st.selectbox("Please select the file to choose", (['None']+files_name))
                    if dataset_dropdown !='None':
                        df_dateset = pd.read_csv(os.path.join(dataset_path,dataset_dropdown))
                        st.write(df_dateset)
                        if st.checkbox('Click here to proceed'):
                            dataset_flag = 1
                    else:
                        dataset_flag = 0
            ############################################################################################
            if (not data) and (dataset_flag == 0):
                show_file.info("Please upload a file of type: " + ", ".join(file_types))
                if os.path.exists(datafile_path):
                    os.remove(datafile_path)
                return
            ############################################################################################
            if (data) or (dataset_flag != 0):
                ############################################################################################
                if st.button("Click to delete data"):
                    if os.path.exists(datafile_path):
                        os.remove(datafile_path)
                        st.success('Raw File deleted successfully!')
                    elif os.path.exists(modifiedfile_path):
                        os.remove(modifiedfile_path)
                        st.success('Modified File deleted successfully!')
                    else:
                        st.markdown(
                            '<span class="badge badge-pill badge-danger"> No Files available for deletion! </span>',
                            unsafe_allow_html=True
                        )
                    # static_store.clear()
                    data = None
                    # session.run_id += 1
                    return
                ############################################################################################
                if (data is not None) or (dataset_flag != 0):
                    if (dataset_flag != 0):
                        df = df_dateset.copy()
                    else:
                        df = pd.read_csv(data)

                    df.to_csv(datafile_path, index=False)
                    st.success('File loaded successfully!')
                ############################################################################################
                if st.checkbox("Click to view data"):
                    if (data is not None) or (dataset_flag != 0):
                        st.write(df)

                        st.write('')
                        st.success('Please proceed to other options from the Activities drop down menu on the Top!')
                    else:
                        #st.write('No Data available!')
                        st.markdown(
                            '<span class="badge badge-pill badge-danger"> No Data available! </span>',
                            unsafe_allow_html=True
                        )
        ############################################################################################
        if choice_1 == "2. Data Summary":
            summary_data(load_data())
        ############################################################################################
        if choice_1 == "3. Data Preprocess":
            preprocess_data(load_data())
        ############################################################################################
        if choice_1 == "4. Data Visualization":
            visualization_data(load_modified_data())
            if st.button('Click here for some Fun!'):
                for i in range(5):
                    st.balloons()
                    time.sleep(1)
        ############################################################################################
    ############################################################################################
    if choice_1_outer == "About":
        st.sidebar.header("About App")
        st.sidebar.info("Complete Data Profiling Application ")
        st.title("")
        st.title("")
        st.sidebar.header("About Developer")
        st.sidebar.info("https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.subheader("About Me")
        st.text("Name: Ashish Gopal")
        st.text("Job Profile: Data Scientist")
        IMAGE_URL = "https://avatars0.githubusercontent.com/u/36658472?s=460&v=4"
        st.image(IMAGE_URL, use_column_width=True)
        st.markdown("LinkedIn: https://www.linkedin.com/in/ashish-gopal-73824572/")
        st.markdown("GitHub: https://github.com/ashishgopal1414")
        st.write('---------------------------------------------------')
    ############################################################################################
################################################### Calling Mains ######################################################
if __name__ == '__main__':
    main()
################################################### END ################################################################