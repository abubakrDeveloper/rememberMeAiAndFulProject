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
      enum:["Teacher", "Student", "Admin", "SupperAdmin"]
    },

    faceImage: {
      type: String
    },
},
{ timestamps: true }
)