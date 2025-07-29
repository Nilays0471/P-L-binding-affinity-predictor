const mongoose = require("mongoose");

const historySchema = new mongoose.Schema({
  userId: { type: String, required: true },
  inputMol: { type: String, required: true },
  affinity: { type: Number, required: true },
  createdAt: { type: Date, default: Date.now },
});

module.exports = mongoose.model("History", historySchema);
