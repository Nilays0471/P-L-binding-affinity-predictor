// server.js
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import admin from "firebase-admin";

admin.initializeApp({ /* service account JSON */ });
//const 
app = express();
app.use(cors(), express.json());
mongoose.connect(process.env.MONGO_URI);

const History = mongoose.model("History", new mongoose.Schema({
  userId: String, inputMol: String, affinity: Number, createdAt: Date
}));

const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const History = require("./models/History");
const verifyFirebaseToken = require("./middleware/verifyToken");

const app = express();
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect("mongodb://localhost:27017/your-db-name", {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("MongoDB connected"))
.catch((err) => console.error("MongoDB connection error:", err));

// POST: Save Prediction
app.post("/api/history", verifyFirebaseToken, async (req, res) => {
  try {
    const { inputMol, affinity } = req.body;
    const record = await History.create({
      userId: req.user.uid,
      inputMol,
      affinity,
      createdAt: new Date(),
    });
    res.json(record);
  } catch (err) {
    console.error("Save error:", err);
    res.status(500).json({ error: "Server error" });
  }
});

// GET: Fetch User History
app.get("/api/history", verifyFirebaseToken, async (req, res) => {
  try {
    const history = await History.find({ userId: req.user.uid }).sort({ createdAt: -1 });
    res.json(history);
  } catch (err) {
    console.error("Fetch error:", err);
    res.status(500).json({ error: "Server error" });
  }
});

// Server listening
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
