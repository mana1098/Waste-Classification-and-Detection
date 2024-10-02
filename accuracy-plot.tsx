import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const data = [
  { epoch: 1, training: 0.65, validation: 0.60 },
  { epoch: 5, training: 0.75, validation: 0.72 },
  { epoch: 10, training: 0.82, validation: 0.78 },
  { epoch: 15, training: 0.87, validation: 0.84 },
  { epoch: 20, training: 0.90, validation: 0.88 },
  { epoch: 25, training: 0.92, validation: 0.91 },
];

const AccuracyPlot = () => (
  <ResponsiveContainer width="100%" height={300}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="epoch" label={{ value: 'Epochs', position: 'insideBottom', offset: -5 }} />
      <YAxis label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
      <Tooltip />
      <Legend />
      <Line type="monotone" dataKey="training" stroke="#8884d8" name="Training Accuracy" />
      <Line type="monotone" dataKey="validation" stroke="#82ca9d" name="Validation Accuracy" />
    </LineChart>
  </ResponsiveContainer>
);

export default AccuracyPlot;
