const User = require("../model/userModel");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

// get All users

const userCtrl = {
  getAllUsers: async (req, res) => {
    try {
    } catch (error) {
      res.status(400).json({ msg: error.message });
    }
  },

  getuser: async (req, res) => {
    try {
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  },

  deleteUser: async (req, res) => {
    try {
    } catch (error) {
      res.status(400).send({ message: error.message });
    }
  },

  updateUser: async (req, res) => {
    try {
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  },
  historyUser: async (req, res) => {
    try {
    } catch (error) {
      res.status(400).json({ message: error.message });
    }
  }
};

module.exports = userCtrl;