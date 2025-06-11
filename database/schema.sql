-- Petstore Database Schema and Sample Data
-- This mirrors the Swagger Petstore API structure

-- Categories table
CREATE TABLE categories (
    id INT PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

-- Pets table
CREATE TABLE pets (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category_id INT,
    status VARCHAR(50) CHECK (status IN ('available', 'pending', 'sold')),
    photo_urls TEXT,
    tags TEXT,
    FOREIGN KEY (category_id) REFERENCES categories(id)
);

-- Users table
CREATE TABLE users (
    id BIGINT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    email VARCHAR(255),
    password VARCHAR(255),
    phone VARCHAR(255),
    user_status INT DEFAULT 1
);

-- Orders table
CREATE TABLE orders (
    id BIGINT PRIMARY KEY,
    pet_id BIGINT,
    user_id BIGINT,
    quantity INT DEFAULT 1,
    ship_date TIMESTAMP,
    status VARCHAR(50) CHECK (status IN ('placed', 'approved', 'delivered')),
    complete BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (pet_id) REFERENCES pets(id),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Insert sample categories
INSERT INTO categories (id, name) VALUES
(1, 'Dogs'),
(2, 'Cats'),
(3, 'Birds'),
(4, 'Fish'),
(5, 'Reptiles');

-- Insert sample pets
INSERT INTO pets (id, name, category_id, status, photo_urls, tags) VALUES
(1, 'Buddy', 1, 'available', 'https://example.com/buddy.jpg', 'friendly,trained'),
(2, 'Max', 1, 'available', 'https://example.com/max.jpg', 'playful,young'),
(3, 'Whiskers', 2, 'pending', 'https://example.com/whiskers.jpg', 'calm,indoor'),
(4, 'Luna', 2, 'available', 'https://example.com/luna.jpg', 'active,outdoor'),
(5, 'Charlie', 3, 'sold', 'https://example.com/charlie.jpg', 'singing,colorful'),
(6, 'Bella', 1, 'available', 'https://example.com/bella.jpg', 'gentle,large'),
(7, 'Milo', 2, 'available', 'https://example.com/milo.jpg', 'curious,small'),
(8, 'Rocky', 1, 'pending', 'https://example.com/rocky.jpg', 'energetic,medium'),
(9, 'Nemo', 4, 'available', 'https://example.com/nemo.jpg', 'tropical,colorful'),
(10, 'Shadow', 2, 'sold', 'https://example.com/shadow.jpg', 'quiet,black');

-- Insert sample users
INSERT INTO users (id, username, first_name, last_name, email, password, phone, user_status) VALUES
(1, 'john_doe', 'John', 'Doe', 'john.doe@email.com', 'password123', '+1-555-0101', 1),
(2, 'jane_smith', 'Jane', 'Smith', 'jane.smith@email.com', 'securepass456', '+1-555-0102', 1),
(3, 'bob_wilson', 'Bob', 'Wilson', 'bob.wilson@email.com', 'mypass789', '+1-555-0103', 1),
(4, 'alice_brown', 'Alice', 'Brown', 'alice.brown@email.com', 'alicepass', '+1-555-0104', 1),
(5, 'charlie_davis', 'Charlie', 'Davis', 'charlie.davis@email.com', 'charliepass', '+1-555-0105', 1),
(6, 'diana_moore', 'Diana', 'Moore', 'diana.moore@email.com', 'dianapass', '+1-555-0106', 1),
(7, 'eve_taylor', 'Eve', 'Taylor', 'eve.taylor@email.com', 'evepass', '+1-555-0107', 1),
(8, 'frank_white', 'Frank', 'White', 'frank.white@email.com', 'frankpass', '+1-555-0108', 1),
(9, 'grace_lee', 'Grace', 'Lee', 'grace.lee@email.com', 'gracepass', '+1-555-0109', 1),
(10, 'henry_clark', 'Henry', 'Clark', 'henry.clark@email.com', 'henrypass', '+1-555-0110', 1);

-- Insert sample orders
INSERT INTO orders (id, pet_id, user_id, quantity, ship_date, status, complete) VALUES
(1, 5, 1, 1, '2024-12-01 10:00:00', 'delivered', TRUE),
(2, 10, 2, 1, '2024-12-02 11:30:00', 'approved', FALSE),
(3, 3, 3, 1, '2024-12-03 14:15:00', 'placed', FALSE),
(4, 8, 4, 1, '2024-12-04 09:45:00', 'placed', FALSE),
(5, 1, 5, 1, '2024-12-05 16:20:00', 'approved', FALSE),
(6, 2, 6, 1, '2024-12-06 13:10:00', 'placed', FALSE),
(7, 4, 7, 1, '2024-12-07 15:30:00', 'delivered', TRUE),
(8, 6, 8, 1, '2024-12-08 12:00:00', 'approved', FALSE),
(9, 7, 9, 1, '2024-12-09 10:45:00', 'placed', FALSE),
(10, 9, 10, 1, '2024-12-10 17:15:00', 'placed', FALSE);