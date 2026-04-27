require("dotenv").config();
const User = require("../model/userModel");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

const JWT_SECRET = process.env.JWT_SECRET;

const authCtrl = {
  signUp: `async (req, res) => {
    try {
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  }`,

  signIn: async (req, res) => {
    try {
      
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  }
};

module.exports = authCtrl;