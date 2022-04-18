from __future__ import print_function

import datetime
import os.path
import logging

logger = logging.getLogger(__name__)

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE_PATH = '/cs/labs/adiyoss/moshemandel/bandwidth-extension/gcp-sheets-service-account-key.json'

SPREADSHEET_ID = '15aGUOUjChziDC-jNYRkZqDSxySeTpjzmGipoIWKJVCI'

def init_creds():
    creds = None
    if os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
        creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE_PATH, scopes=SCOPES)
    else:
        logger.info(f'SERVICE_ACCOUNT_FILE_PATH not correct.current path:{os.getcwd()}')
    return creds

def log(experiment_name, metrics, lr_sr, hr_sr):
    creds = init_creds()
    sheet_name = str(lr_sr).strip('0') + '->' + str(hr_sr).strip('0')
    range_name = f'{sheet_name}!A2:A'
    timd_str = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    try:
        service = build('sheets', 'v4', credentials=creds)
        value_range_body = {
            "range": range_name,
            "majorDimension": 'ROWS',
            "values": [
                [timd_str, experiment_name,
                 metrics['pesq'], metrics['stoi'], metrics['sisnr'], metrics['visqol'], metrics['lsd']]
            ]
        }

        request = service.spreadsheets().values().append(spreadsheetId=SPREADSHEET_ID, range=range_name,
                                                         valueInputOption='RAW',
                                                         insertDataOption='OVERWRITE', body=value_range_body)
        response = request.execute()

    except HttpError as err:
        print(err)


def main():
    metrics = {'pesq':1, 'stoi': 2, 'sisnr': 3, 'visqol': 4, 'lsd': 5}
    experiment_name = 'test'
    log(experiment_name, metrics, 8000, 16000)

if __name__ == '__main__':
    main()