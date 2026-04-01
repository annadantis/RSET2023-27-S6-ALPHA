-- MySQL Database Setup Script for Placement App
-- Run this script to create the database and tables

CREATE DATABASE IF NOT EXISTS placement_app;
USE placement_app;

-- Table for user profiles
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    aptitude_level INT DEFAULT 1,
    technical_level INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for historical performance
CREATE TABLE IF NOT EXISTS results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    score INT NOT NULL,
    area VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_category (category),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for questions (for the import script)
CREATE TABLE IF NOT EXISTS questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    option_a TEXT,
    option_b TEXT,
    option_c TEXT,
    option_d TEXT,
    correct_answer TEXT,
    category VARCHAR(50),
    area VARCHAR(100),
    difficulty VARCHAR(20),
    explanation TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for GD evaluations
CREATE TABLE IF NOT EXISTS gd_evaluations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255),
    topic TEXT,
    transcript TEXT,
    content_score INT,
    communication_score INT,
    feedback TEXT,
    audio_path VARCHAR(500),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table for branch-specific interview questions
CREATE TABLE IF NOT EXISTS interview_questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    branch VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    ideal_answer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_branch (branch)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table to track which user has been asked which interview question
CREATE TABLE IF NOT EXISTS user_asked_questions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    question_id INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;