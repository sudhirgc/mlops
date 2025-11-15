import mlflow
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import pandas as pd
import yfinance
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import pickle
import os
import shutil
import optuna

logger = logging.getLogger(__name__)
scaler = MinMaxScaler()

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

class LSTM(nn.Module):
    def __init__(self, no_features:int, no_neurons:int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=no_features,hidden_size=no_neurons)
        self.regressor = nn.Sequential(
            nn.Linear(in_features=no_neurons,out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32,out_features=1)
        )
    
    def forward(self, X):
        X,_ = self.lstm(X)
        print(f'Forward Method -> {X.shape}')
        X = self.regressor(X[:,-1,:])
        return X

def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: # Hidden-hidden weights
                nn.init.orthogonal_(param.data) # Orthogonal initialization often works well for recurrent weights
            elif 'bias' in name:
                param.data.fill_(0)
                # Initialize forget gate bias to a higher value (e.g., 1) to promote long-term memory
                # This is a common practice for LSTMs
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

def create_TS_dataset(df, time_series_length):
    X,y = [], []
    for i in range(len(df) - time_series_length):
        X_iter = df[i:time_series_length+i]
        y_iter = df[i+1:time_series_length+i+1]
        X.append(X_iter)
        y.append(y_iter)
    #print('X -> ',X)
    #print('y -> ',y)
    return torch.tensor(X), torch.tensor(y)

def create_last_tensor(df, time_series_length):
    df = scaler.fit_transform(df[['Close']])
    X = []
    X.append(df[-time_series_length:])
    return torch.tensor(X).type(torch.float32)

def accuracy_precent(y_true, y_pred):
    y_pred[y_pred == 0] = np.nan
    y_true[y_true == 0] = np.nan
    acc = torch.nanmean(100 - abs((y_true - y_pred)/y_true)*100) 
    print("Accuracy ->", acc.item())
    return acc.item()

def analyse_stock_performance(ticker:str):
    df = get_ticker_data(ticker)

    """ fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df.index.values, df[['Open','High','Low','Close']].values)
    ax.legend(loc="best")  """

    model, all_values, accuracy = create_model_and_generate_predictions(ticker, df)

    delete_dir(params['output_directory'])
    os.makedirs(params['output_directory'], exist_ok=True)
    mlflow.pytorch.save_model(model,path=params['output_directory'])
    mlflow.pytorch.log_model(model,name=params['model_name'])

def create_model_and_generate_predictions(ticker, df):
    delete_dir("data")
    os.makedirs(f"data", exist_ok=True)

    df.to_csv(f"data/{ticker}_dataframe.csv")
    mlflow.log_artifact(f"data/{ticker}_dataframe.csv", "data")

    #time_series = df[['Open','High','Low','Close']].values.astype('float32')
    time_series = df[['Close']].values.astype('float32')
    train_loader, test_loader = create_dataloaders(time_series)
    
    model = LSTM(no_features=params['no_features'],no_neurons=params['no_neurons'])
    model.apply(init_weights)
    model.to(device=params["device"])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=params['learning_rate'])
    early_stopping = EarlyStopping(patience=5, delta=0.005)

    torch.cuda.empty_cache()
    accuracy = 0.0
    for epoch in range(params["num_epochs"]):
        train_model(train_loader, model, loss_fn, optimizer, epoch)
        break_early_stopping, accuracy = eval_model(test_loader, model, loss_fn, epoch, early_stopping=early_stopping)
        if break_early_stopping == True:
            early_stopping.load_best_model(model)
            break

    y_pred_list = predict_with_model(time_series, model)
    print('Predictions Horizons -> ', y_pred_list)

    all_values = []
    all_values.extend(df['Close'].values)
    all_values.extend(y_pred_list) 
    print('All Values -> ', all_values)

    plt.plot(all_values)
    return model, all_values, accuracy

def get_ticker_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    ticker_obj = yfinance.Ticker(ticker=ticker)

    df = ticker_obj.history(start=start_date, end=end_date)
    print(df.head())
    print(df.info())
    df.dropna(inplace=True)
    return df
    

def delete_dir(directory_to_delete):
    if os.path.isdir(directory_to_delete):
        try:
            shutil.rmtree(directory_to_delete)
            print(f"Directory '{directory_to_delete}' and its contents deleted successfully.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}.")
    else:
        print(f"Directory '{directory_to_delete}' does not exist.")

def predict_with_model(time_series, model):
    y_pred_list = []
    X_for_prediction = create_last_tensor(time_series, params["sequence_length"])
    print('X_Pred', X_for_prediction, 'Shape ->', X_for_prediction.shape)
    for i in range(params['forecast_horizon']):
        y_pred = model(X_for_prediction.to(device=device))
        y_pred = y_pred.cpu()
        
        print('Y_Pred -> ', y_pred, 'Shape ->', y_pred.shape)
        X_for_prediction = torch.cat((X_for_prediction[:,1:,:], torch.unsqueeze(y_pred,dim=1) ), dim=1) 
        print('X_Pred Updated -> ', X_for_prediction, 'Shape ->', X_for_prediction.shape)

        y_pred_list.append(scaler.inverse_transform(y_pred.detach()).squeeze().squeeze())

    return y_pred_list

def create_dataloaders(time_series):
    time_series = scaler.fit_transform(time_series)
    
    # Lets split the time_series into test, train datasets 
    X, y = create_TS_dataset(time_series, time_series_length=params["sequence_length"]) 
    print(' Features -> ',X.shape)
    print(' Target -> ',y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_ratio"], random_state=42)
    print(' Train Features -> ',X_train.shape, 'Test Features -> ', X_test.shape)
    print(' Train Target -> ',y_train.shape, 'Test Target -> ', y_test.shape)
    
    train_loader = DataLoader(TensorDataset(X_train,y_train),shuffle=True,batch_size=params["batch_size"])
    test_loader = DataLoader(TensorDataset(X_test,y_test),shuffle=True,batch_size=params["batch_size"])
    return train_loader,test_loader

def eval_model(test_loader, model, loss_fn, epoch, early_stopping):
    model.eval()
    test_loss = 0
    test_accuracy = []
    accuracy = 0.0
    #Lets evaluate the Model and find out test accuracy
    with torch.inference_mode():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)
            output = model(inputs)
            print('input ->', inputs)
            print('output ->', output)
            print('labels ->',labels[:,-1,:])
            loss = loss_fn(output, labels[:,-1,:])
            print('>>>> LOSS ->>> ',loss)
            test_loss+=loss
            accuracy_perc = accuracy_precent(labels[:,-1,:].cpu().detach().squeeze(), output.cpu().detach().squeeze())
            test_accuracy.append(accuracy_perc)
            
        if epoch % 1 == 0:
            print(test_accuracy)
            accuracy =  np.mean(test_accuracy, axis=0)
            print(f" Test Loss {test_loss/len(test_loader)}, Test Accuracy {np.mean(test_accuracy, axis=0)}")
            mlflow.log_metrics({'accuracy': accuracy, 'loss': test_loss/len(test_loader)}, step=epoch)
            
        early_stopping(test_loss/len(test_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True, accuracy

    return False, accuracy
        

def train_model(train_loader, model, loss_fn, optimizer, epoch):
    model.train()
    training_loss = 0.0
    training_accuracy_perc = []
        
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch

        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        output = model(inputs)

        print('output ->', output)
        print('labels ->',labels[:,-1,:])

        loss = loss_fn(output, labels[:,-1,:])

        print('>>>> LOSS ->>> ',loss)
        loss.backward()
        optimizer.step()

        training_loss+=loss
        accuracy_perc = accuracy_precent(labels[:,-1,:].cpu().detach().squeeze(), output.cpu().detach().squeeze())
        training_accuracy_perc.append(accuracy_perc)
        #Lets print the training_loss and MAEP
    if epoch % 1 == 0:
        print(training_accuracy_perc)
        print(f" Training Loss {training_loss/len(train_loader)}, Training Accuracy {np.mean(training_accuracy_perc, axis=0)}")
        mlflow.log_metrics({'accuracy': np.mean(training_accuracy_perc, axis=0), 'loss': training_loss/len(train_loader)}, step=epoch)


device = 'cpu'
if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

params = {
    "model_name":"LSTM_1D_Hyper_2",
    "learning_rate" : 0.0005,
    "batch_size":8,
    "num_epochs":30,
    "dataset_name":"yfinance_history",
    "task_name": "regression_hyperparam_5",
    "sequence_length": 30,
    "forecast_horizon":6,
    "device":device,
    "output_directory":"models/LSTM_1D_Hyper_2_yfinance",
    "no_features":1,
    "no_neurons":100,
    "test_ratio":0.2
}

ticker = 'AAPL'

def objective(trial):
    #Get data once..
    df = get_ticker_data(ticker=ticker)
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        # Hyper params - batch_size, training_rate, number_epochs, no_neurons
        rf_batch_size = trial.suggest_int("rf_batch_size",8,32,step=8)
        rf_num_epochs = trial.suggest_int("rf_num_epochs",20,50,step=10)
        rf_learning_rate = trial.suggest_float("rf_learning_rate",0.0005,0.01,step=0.0005)
        params['batch_size'] = rf_batch_size
        params['num_epochs'] = rf_num_epochs
        params['learning_rate'] = rf_learning_rate

        mlflow.log_params(params)
        model,all_values, accuracy = create_model_and_generate_predictions(ticker=ticker,df=df)
        mlflow.pytorch.log_model(model,name="model")     
        trial.set_user_attr("run_id", child_run.info.run_id)
        return accuracy

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(params['task_name'])

os.makedirs(params['output_directory'], exist_ok=True)

def run_single_flow(scaler, params, ticker):
    with mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}") as run:
        mlflow.log_params(params)
        analyse_stock_performance(ticker)

        with open(f'{params["output_directory"]}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        mlflow.log_artifact(f'{params["output_directory"]}/scaler.pkl')
        mlflow.log_artifacts(params['output_directory'], artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, params['model_name'])

# run_single_flow(scaler, params, ticker)

def run_hyperparam_flow(params, objective):
    with mlflow.start_run(run_name=f"{params['model_name']}-{params['dataset_name']}-study") as run:
        n_trials = 80
        params['task_name'] = "regression_hyperparam_5"
        mlflow.log_param("n_trials", n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective,n_trials=n_trials)
        mlflow.log_params(study.best_trial.params)
        #mlflow.log_metrics({"best_accuracy",study.best_value})

        if best_run_id := study.best_trial.user_attrs.get("run_id"):
            mlflow.log_param("best_child_run_id", best_run_id)

            mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=params['model_name'],
        )

        delete_dir(params['output_directory'])
        os.makedirs(params['output_directory'], exist_ok=True)
        with open(f'{params["output_directory"]}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        mlflow.log_artifact(f'{params["output_directory"]}/scaler.pkl',"model")

#run_hyperparam_flow(params, objective)

# Load a specific model version
def download_eval_model(get_ticker_data, predict_with_model, params):
    model_name = params['model_name']
    model_version = "1"  # or "production", "staging"


    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pytorch.load_model(model_uri)

    print(model)

    """ run_id = "88802afe387b4e4b99cd1505ce1eac0b"  # Replace with the actual run ID
    artifact_path = "model/scaler.pkl"  # Replace with the actual artifact path
    pickle_uri = f"runs:/{run_id}/{artifact_path}" 
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=pickle_uri) """
    local_path = f"{params['output_directory']}/scaler.pkl"
    with open(local_path, "rb") as f:
        scaler = pickle.load(f)

    print(predict_with_model(get_ticker_data('AAPL'),model=model))

download_eval_model(get_ticker_data, predict_with_model, params)


