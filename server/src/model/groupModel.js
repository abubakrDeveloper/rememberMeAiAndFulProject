const mongoose = require("mongoose");

const groupSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      unique: true,
      trim: true,
      uppercase: true,
    },
    department: {
      type: String,
      required: true,
    },

    course: {
      type: Number,
      min: 1,
      max: 4,
      required: true,
    },

    status: {
      type: String,
      enum: ["Active", "Graduated", "Inactive"],
      default: "Active",
    },
  },
  { timestamps: true },
);

module.exports = mongoose.model("groupSchema", groupSchema);