
from database import (
    create_tables,
    save_prediction,
    get_prediction_stats
)

create_tables()

record_id = save_prediction(1, 0.82)

print("Saved prediction ID:", record_id)
print("Statistics:", get_prediction_stats())
