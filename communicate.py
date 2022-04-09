from twilio.rest import Client 
 
account_sid = 'AC2b374f6b60483ce50c017c305818343a' 
auth_token = '11ae8cf073d68df9a9488ea49c8af316' 
client = Client(account_sid, auth_token) 

def Send_Warning():
 message = client.messages.create(  
                              messaging_service_sid='MGa281841b121b9b99a83546dcc8c03af9', 
                              body='Please Turn off the Lights. Electricity is being wasted!',      
                              to='+917667346707' 
                          ) 
 
 print(message.sid)