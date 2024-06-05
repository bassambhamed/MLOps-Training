from pydantic import BaseModel, Field
from pydantic import BaseModel
import datetime



class TransactionModel(BaseModel):
   trans_date_trans_time : datetime = Field(...)
   category :object = Field(...)
   amt : float = Field(...)
   gender : object = Field(...)
   zip : int = Field(...)
   lat  : float = Field(...)
   long  : float = Field(...)
   dob : object = Field(...)
   merch_lat  : float = Field(...)
   merch_long : float = Field(...)

   class Config:
       populate_by_name = True
       arbitrary_types_allowed = True
       json_schema_extra = {
           "example": {
               "trans_date_trans_time": "2019-04-13 08:32:53",
               "category" : "misc_net",
               "amt": "4.97",
               "gender": 'F',
               "zip": "17060",
               "lat": "36.0788" ,
               "long": "-81.1781",
               "dob": "1988-03-09",
               "merch_lat": "36.011293",
               "merch_long": "-82.048315"

           }
       }