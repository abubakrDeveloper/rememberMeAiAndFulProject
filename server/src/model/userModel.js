const mongoose = require("mongoose");

const userSchema =  new mongoose.Schema({
    firstName: {
      type: String, 
      requried: true,
    },

    lastName: {
      type: String, 
      required: true
    },

    phoneNumber: {
      type: String, 
      required: true
    },

    password: {
      type: String
    },

    role: {
      type: String, 
      default: "student", 
      enum:["Teacher", "Student", "Admin", ]
    },

    faceImage: {
      type: String
    },
},
{ timestamps: true }
)

module.exports = mongoose.model("userSchema", userSchema);