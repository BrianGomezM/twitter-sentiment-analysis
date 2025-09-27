from sqlalchemy import create_engine

def get_engine():
    # 🔹 Para PostgreSQL en Render
    engine = create_engine(
        "postgresql+psycopg2://redes_neuronales_proyecto_user:"
        "qgZaEQHbnkqio5wojYT9VldBH81XYn1k@"
        "dpg-d3bmi5ggjchc738ij1m0-a.oregon-postgres.render.com:5432/"
        "redes_neuronales_proyecto"
    )
    return engine
