const express = require('express');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const cors = require('cors');
const bodyParser = require('body-parser');

dotenv.config();

const app = express();

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use('/uploads', express.static('uploads')); // Rasmlarni saqlash uchun

mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/smart-attendance', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log("MongoDB muvaffaqiyatli ulandi"))
.catch((err) => console.error("MongoDB ulanishida xato:", err));

app.get('/', (req, res) => {
    res.send("Smart Attendance API tizimi ishlamoqda...");
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(` Server ${PORT}-portda ishga tushdi`)
});