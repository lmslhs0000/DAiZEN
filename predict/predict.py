import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

cnn_model = tf.keras.models.load_model('cnn_model.keras')

test_df = pd.read_csv('raw/ex_test.csv')

kernel_width = 3


def recursive_forecast(model, data, steps=24, kernel_width=3):

    history = data.tail(kernel_width).copy()
    predictions = []

    for step in range(steps):

        x_input = history.tail(kernel_width).values.astype(np.float32)
        x_input = x_input[np.newaxis, :, :]

        pred = model.predict(x_input, verbose=0)

        next_no2 = float(pred[0, 0, 0])

        predictions.append(next_no2)

        next_row = history.iloc[-1].copy()
        next_row['NO2'] = next_no2

        current_sin = history.iloc[-1]['day_sin'] * 2 - 1
        current_cos = history.iloc[-1]['day_cos'] * 2 - 1

        current_angle = np.arctan2(current_sin, current_cos)
        next_angle = current_angle + (2 * np.pi / 24)

        next_row['day_sin'] = (np.sin(next_angle) + 1) / 2
        next_row['day_cos'] = (np.cos(next_angle) + 1) / 2

        history = pd.concat(
            [history, next_row.to_frame().T],
            ignore_index=True
        )

    return np.array(predictions)


def predict(data, steps=24):

    future = recursive_forecast(
        model=cnn_model,
        data=data,
        steps=steps,
        kernel_width=kernel_width
    )

    forecast_df = pd.DataFrame({
        'hour_ahead': range(1, steps + 1),
        'NO2_prediction_scaled': future
    })

    forecast_df.to_csv(
        'future_24_no2.csv',
        index=False
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        range(1, steps + 1),
        future,
        marker='o'
    )

    plt.xlabel('Future Time (h)')
    plt.ylabel('NO2 [scaled]')
    plt.title('NO2 Future 24-Hour Forecast')

    plt.xticks(range(1, steps + 1))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('future_24_no2.png')
    plt.close()

    result = {
        "predictions": future.tolist(),
         "target": "NO2"
    }

    return result

result = predict(test_df)
print(result)