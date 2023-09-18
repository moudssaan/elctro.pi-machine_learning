import pandas as pd
import streamlit as st
import plotly_express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


if os.path.exists('dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar:
    choice = st.radio("Pleas follow the steps below in order:", ["Uploading Data","Applying Exploratory Data Analysis",'Preprocessing Data',"Modeling Phase"])


if choice == "Uploading Data":
    st.title("Uploading Data")
    file = st.file_uploader("Upload your .csv file")

    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.write(df.shape)

#######################################################################################################################################
if choice == "Applying Exploratory Data Analysis":
    st.title("Applying Exploratory Data Analysis")
    
    eda_choise = st.selectbox('Select an analysis Operation',['','Show Data Types','Show Null Values','Show Numerical features descriptions','Show Value Counts'])

    if eda_choise =='Show Data Types' :
        st.write(df.dtypes)

    if eda_choise == 'Show Null Values' :
        st.write(df.isna().sum())

    if eda_choise =='Show Numerical features descriptions' :
        st.write(df.describe())

    if eda_choise =='Show Value Counts' :
        try:
            selected_columns = st.multiselect('Select desired columns', df.columns)
            new_df = df[selected_columns]
            st.write(new_df.value_counts().rename(index='Value'))

        except:
            pass


    plot_choice = st.selectbox('Select an analysis plot',['','Box Plot','Scatter Plot','Bar Plot', 'Histogram'])


    if plot_choice == 'Box Plot' :
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        fig = px.box(df,y=column_to_plot)
        st.plotly_chart(fig)

    if plot_choice =='Scatter Plot' :

        try :
            selected_columns = st.multiselect('Select two columns',df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.scatter(df, x=first_column, y=second_column)
            fig.update_layout(title="Scatter Plot", xaxis_title=first_column, yaxis_title=second_column)
            st.plotly_chart(fig)

        except:
            pass

    if plot_choice == 'Bar Plot':
        try :
            selected_columns = st.multiselect('Select columns', df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]

            fig = px.bar(df, x=first_column, y=second_column, title='Bar Plot')
            st.plotly_chart(fig)

        except :
            pass

    if plot_choice == 'Histogram':
        try :
            column_to_plot = st.selectbox("Select a Column", df.columns)
            fig = px.histogram(df,x=column_to_plot)
            st.plotly_chart(fig)

        except :
            pass

#######################################################################################################################################
if choice == "Preprocessing Data" :
    st.title('Preprocessing Data')

    st.subheader("Dropping Useless Columns")
    want_to_drop = st.selectbox('Do you want to drop any columns ?',['','Yes','No'])

    if want_to_drop == 'No':
        pass

    if want_to_drop == 'Yes':

        columns_to_drop = st.multiselect('Columns to drop', df.columns)
        if columns_to_drop  :
            df = df.drop(columns_to_drop, axis=1)
            st.success('Columns dropped')
            st.dataframe(df)

    st.subheader("Filling Null Values")    
    fill_option = st.selectbox('Do you want to fill numerical features?', ['', 'Yes', 'No'])

    if fill_option == 'No':
        pass

    if fill_option == 'Yes':

        encoder_columns = st.multiselect('Numerical columns to fill', df.select_dtypes(include=['number']).columns)
        encoder_type = st.selectbox('Imputation Method', ['','Mean','Median'])

        try:
            if encoder_type == 'Mean' :
                imputer = SimpleImputer(strategy='mean')
                df[encoder_columns] = imputer.fit_transform(df[encoder_columns])
                st.success('Numerical columns filled')
                st.dataframe(df)

            if encoder_type == 'Median' :
                imputer = SimpleImputer(strategy='median')
                df[encoder_columns] = imputer.fit_transform(df[encoder_columns])
                st.success('Numerical columns filled')
                st.dataframe(df)

        except :
            pass

    fill_option2 = st.selectbox('Do you want to fill categorical features?', ['', 'Yes', 'No'])

    if fill_option2 == 'No':
        pass

    if fill_option2 == 'Yes':

        encoder_columns = st.multiselect('Categorical columns to fill', df.select_dtypes(include=['object']).columns)
        encoder_type = st.selectbox('Imputation Method', ['','Most frequent'])

        if encoder_type == 'Most frequent' :

            imputer = SimpleImputer(strategy='most_frequent')
            df[encoder_columns] = imputer.fit_transform(df[encoder_columns])
            st.success('Categorical columns filled')
            st.dataframe(df)

    st.subheader("Encoding Categorical Features")
    encoder_option = st.selectbox('Do you want to encode your categorical features ?',['','Yes','No'])

    if encoder_option == 'No' :
        pass

    if encoder_option == 'Yes' :

        encoder_columns = st.multiselect('Columns to encode',df.select_dtypes(include=['object']).columns)
        encoder_type = st.selectbox('Select an encoder', ['','Label Encoder','One Hot Encoder'])

        if encoder_type == 'Label Encoder' :

            encoder = LabelEncoder()
            df[encoder_columns] = df[encoder_columns].apply(encoder.fit_transform)
            st.success('Columns encoded.')
            st.dataframe(df)

        if encoder_type == 'One Hot Encoder':

            df = pd.get_dummies(df, columns=encoder_columns, prefix=encoder_columns,drop_first=True)
            st.success('Columns encoded.')
            st.dataframe(df)


    st.subheader("Scaling Numerical Features")
    scaling_option = st.selectbox('Do you want to scale your numerical features?',['','Yes','No'])

    if scaling_option == 'No' :
        st.dataframe(df)
        df = df.to_csv('dataset.csv', index=None)

    if scaling_option == 'Yes' :

        scaling_method = st.selectbox("Scaling Method", ('','Standardization', 'Min-Max'))

        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if scaling_method == 'Standardization':
            scaler = StandardScaler()
            df_scaled = df.copy()  # Create a copy of the original DataFrame
            df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            st.success('The numerical features has been scaled')
            st.dataframe(df_scaled)
        
        elif scaling_method == 'Min-Max':
            scaler = MinMaxScaler()
            df_scaled = df.copy()  # Create a copy of the original DataFrame
            df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            st.success('The numerical features has been scaled')
            st.dataframe(df_scaled)

        try :
            df = df_scaled.to_csv('dataset.csv', index=None)
        except :
            pass

#######################################################################################################################################
if choice == "Modeling Phase":

    st.title('Modeling Phase')
    df = pd.read_csv('dataset.csv', index_col=None)

    target_choices = [''] + df.columns.tolist()

    st.subheader("Splitting Data & Choosing Target")
    try :
        target = st.selectbox('Target variable', target_choices)
        X = df.drop(columns=target)
        y = df[target]

        test_size = st.select_slider('Specify the size of the testing set', range(1, 100, 1))
        test_size_fraction = test_size / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)
        st.write('Shape of training data is :', X_train.shape)
        st.write('Shape of testing data is :', X_test.shape)

    except :
        pass

    st.subheader("Modeling")
    task_type = st.selectbox('Supervised learning algorithm', ['','Classification', 'Regression'])
    modeling_choice = st.selectbox('Do you want Auto modeling or choose the model ?',
                                   ['','Auto modeling','Manual modeling'])

    if task_type == 'Classification':

        if modeling_choice == 'Auto modeling':
            try:
                if st.button('Go Model'):
                    from sklearn.metrics import accuracy_score
                    accuracy = []
                        
                    #Logistic Regression
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy.append(accuracy_score(y_test, y_pred))

                    #Decision Tree
                    from sklearn.tree import DecisionTreeClassifier
                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy.append(accuracy_score(y_test, y_pred))

                    #KNN
                    from sklearn.neighbors import KNeighborsClassifier
                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                    accuracy.append(accuracy_score(y_test, y_pred))

                    accuracies = {   
                        'Logistic Regression': accuracy[0],
                        'Decision Tree': accuracy[1],
                        'KNN': accuracy[2],
                    }

                    best_model = max(accuracies, key=accuracies.get)
                    st.write("Best Model:", best_model)
            except:
                st.warning('Your Target Variable is Numerical, It should be Categorical')

        
        if modeling_choice == 'Manual modeling' :
            try:
                algo_type = st.selectbox('Choose the classifer',['','Logistic Regression','Decision Trees','KNN'])

                if algo_type == 'Logistic Regression' :
                    from sklearn.linear_model import LogisticRegression

                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)


                if algo_type == 'Decision Trees' :
                    from sklearn.tree import DecisionTreeClassifier

                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'KNN' :
                    from sklearn.neighbors import KNeighborsClassifier

                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                evaluation_type = st.selectbox('Choose type of assessment metrics ',['','Accuracy','Confusion Matrix'])

                if evaluation_type == 'Accuracy' :

                    from sklearn.metrics import accuracy_score

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write("Accuracy:", accuracy)

                if evaluation_type == 'Confusion Matrix' :

                    from sklearn.metrics import confusion_matrix

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(cm)
            except:
                st.warning('Your Target Variable is Numerical, It should be Categorical')                

    if task_type == 'Regression':

        if modeling_choice == 'Auto modeling':
            try:
                if st.button('Go Model'):
                    from sklearn.metrics import r2_score
                    acc = []
                        
                    #Linear Regression
                    from sklearn.linear_model import LinearRegression
                    rg = LinearRegression()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)
                    acc.append(r2_score(y_test, y_pred))

                    #Random Forest
                    from sklearn.ensemble import RandomForestRegressor
                    rg = RandomForestRegressor()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)
                    acc.append(r2_score(y_test, y_pred))

                    accuracies = {   
                        'Linear Regression': acc[0],
                        'Random Forest': acc[1]
                    }

                    best_model = max(accuracies, key=accuracies.get)
                    st.write(f"Best Model: {best_model}")
            except:
                st.warning('Your Target Variable is Categorical, It should be Numerical')

        if modeling_choice == 'Manual modeling' :
            try:
                algo_type = st.selectbox('Choose the regressor',
                                        ['','Linear Regression','Random Forest'])

                if algo_type == 'Linear Regression' :

                    from sklearn.linear_model import LinearRegression

                    rg = LinearRegression()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'Random Forest' :

                    from sklearn.ensemble import RandomForestRegressor

                    rg = RandomForestRegressor()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                evaluation_type = st.selectbox('Choose type of assessment metrics ',['','MAE','MSE','r2 score'])

                if evaluation_type == 'MAE' :

                    from sklearn.metrics import mean_absolute_error

                    MAE = mean_absolute_error(y_test, y_pred)
                    st.write("Mean absolute error:", MAE)

                if evaluation_type == 'MSE' :

                    from sklearn.metrics import mean_squared_error

                    MSE = mean_squared_error(y_test, y_pred)
                    st.write("Mean squared error:", MSE)

                if evaluation_type == 'r2 score' :

                    from sklearn.metrics import r2_score

                    r2 = r2_score(y_test, y_pred)
                    st.write("r2 score:", r2)
            except:
                st.warning('Your Target Variable is Categorical, It should be Numerical')