const mongoose = require("mongoose");

const attendanceSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    // davomat statusi
    status: {
      type: String,
      default: "Present",
      enum: ["Present", "Absent", "Late", "Excused"], // Keldi, Yo'q, Kechikdi, Sababli
    },

    groupNumber: {
      type: String,
      required: true,
    },

    // AI tomonidan olingan rasm (isbot sifatida saqlash uchun)
    captureImage: {
      type: String,
    },

    // Dars soati yoki tartib raqami
    lessonOrder: {
      type: Number,
      enum: [1, 2, 3, 4, 5, 6],
    },
  },
  { timestamps: true },
);

module.exports = mongoose.model("attendanceModel", attendanceModel);
