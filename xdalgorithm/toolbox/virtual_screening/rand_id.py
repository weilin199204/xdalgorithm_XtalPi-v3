import bson

def get_rand_id():
    return str(bson.ObjectId())[::-1]
