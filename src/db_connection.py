import psycopg2

def get_connection():
    conn = psycopg2.connect(
        host="dpg-d3bmi5ggjchc738ij1m0-a.oregon-postgres.render.com",
        user="redes_neuronales_proyecto_user",
        password="qgZaEQHbnkqio5wojYT9VldBH81XYn1k",   # sin contraseña
        database="redes_neuronales_proyecto",
        port=5432
    )
    return conn
