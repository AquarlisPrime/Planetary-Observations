import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
from astropy.coordinates import CartesianRepresentation
from astropy import units as u
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import spacy
from spacy.tokens import Span
from matplotlib.patches import Rectangle
from PIL import Image  # Import PIL to handle image resizing
import mplcursors  # Import mplcursors for cursor annotations

nlp = spacy.load("en_core_web_sm")

def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def analyze_orbital_data(orbital_data, labels):
    cnn_model = build_cnn_model(input_shape=orbital_data.shape[1:])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(orbital_data, labels, epochs=10, batch_size=32)
    return cnn_model

def compute_positions_and_orbits(year):
    positions = {}
    orbits = {}
    
    start_time = Time(f'{year}-01-01', scale='utc')
    end_time = start_time + 365 * u.day  # Simulate one year of data

    with solar_system_ephemeris.set('builtin'):
        planets = ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

        for planet_name in planets:
            times = Time(np.linspace(start_time.jd, end_time.jd, 1000), format='jd')
            planet = get_body_barycentric(planet_name, times)
            positions[planet_name] = planet.represent_as(CartesianRepresentation).xyz.value
            orbits[planet_name] = planet.represent_as(CartesianRepresentation).xyz.value

    return positions, orbits

def update(frame, current_year, positions, orbits, planet_images):
    plt.cla()
    plt.xlabel('X (AU)')
    plt.ylabel('Y (AU)')
    plt.title(f'Planetary Orbits and Positions in {current_year}')
    plt.gca().set_facecolor('black')  # Set background color to black

    for i, (planet_name, orbit) in enumerate(orbits.items()):
        x_orbit, y_orbit, _ = orbit
        x_pos, y_pos, _ = positions[planet_name]

        # Plot full orbit path
        plt.plot(x_orbit, y_orbit, color='white', alpha=0.3)

        # Load planet image and resize
        planet_image_path = planet_images.get(planet_name)
        if planet_image_path:
            planet_image = Image.open(planet_image_path)
            planet_image = planet_image.resize((500, 500))  # Resize image for better visibility

            # Plot the planet image at the current position
            plt.imshow(planet_image, extent=(x_pos[frame] - 0.5, x_pos[frame] + 0.5, y_pos[frame] - 0.5, y_pos[frame] + 0.5))

            # Define a function to handle cursor hover event for this planet
            def on_hover(sel):
                # Check which planet the cursor is hovering over
                if sel.artist.get_label() in planet_images:
                    planet_name = sel.artist.get_label()
                    planet_image_path = planet_images[planet_name]

                    # Load and display the corresponding planet image
                    planet_image = Image.open(planet_image_path)
                    planet_image = planet_image.resize((500, 500))

                    # Display the image in a new figure
                    display_zoomed_image(sel, planet_name, planet_image)

            # Add annotation to display zoomed image and planet name on hover
            mplcursors.cursor(hover=True).connect("add", on_hover)

    # Set aspect ratio to equal and adjust axis limits for the animation space
    plt.gca().set_aspect('equal')
    plt.xlim(-50, 50)  # Adjust x-axis limits for the animation space
    plt.ylim(-50, 50)  # Adjust y-axis limits for the animation space

    plt.show()


def display_zoomed_image(sel, planet_name, planet_image):
    print(f"Hovering over planet: {planet_name}")  # Debug message to check planet name
    fig, ax = plt.subplots()
    ax.imshow(planet_image)
    ax.set_title(planet_name)
    plt.show()

def animate(year):
    positions, orbits = compute_positions_and_orbits(year)
    num_frames = len(next(iter(positions.values()))[0])  # Number of frames based on position data length

    # Load planet images for each planet
    planet_images = {
        'mercury': r"C:\Users\Dell\Desktop\planet imgs\mercury.jpeg",
        'venus': r"C:\Users\Dell\Desktop\planet imgs\venus.jpeg",
        'earth': r"C:\Users\Dell\Desktop\planet imgs\earth.jpeg",
        'mars': r"C:\Users\Dell\Desktop\planet imgs\mars.jpeg",
        'jupiter': r"C:\Users\Dell\Desktop\planet imgs\jupyter.png",
        'saturn': r"C:\Users\Dell\Desktop\planet imgs\saturn.png",
        'uranus': r"C:\Users\Dell\Desktop\planet imgs\uranus.jpeg",
        'neptune': r"C:\Users\Dell\Desktop\planet imgs\neptune.jpg"
    }

    fig = plt.figure(figsize=(10, 10))
    ani = animation.FuncAnimation(fig, update, frames=num_frames, fargs=(year, positions, orbits, planet_images),
                                  interval=50, repeat=False)

    plt.show()

# Testing the animation
animate(2024)  # Adjust the year as needed


def process_user_input(input_text):
    nlp = pipeline("ner", grouped_entities=True)
    entities = nlp(input_text)
    year = None

    for entity in entities:
        if entity['entity'] == 'DATE':
            try:
                year = int(entity['word'])
                break
            except ValueError:
                continue

    return year

def main():
    while True:
        year_input = input("Enter a year between 1900 and 2040: ")

        try:
            year = int(year_input)
            if 1900 <= year <= 2040:
                animate(year)
                break
            else:
                print("Invalid year. Please enter a year between 1900 and 2040.")
        except ValueError:
            print("Invalid input. Please enter a valid year.")

if __name__ == "__main__":
    main()
