import axios from "axios";
import React, { useState } from "react";

function Dashboard() {
  const [file, setFile] = useState(null);
  const [name, setName] = useState("");

  const submit = async () => {
    const form = new FormData();
    form.append("file", file);

    const res = await axios.post(
      "http://127.0.0.1:8000/mark-attendance",
      form
    );
    setName(res.data.name);
  };

  return (
    <div>
      <h2>Face Attendance</h2>
      <input type="file" onChange={e => setFile(e.target.files[0])} />
      <button onClick={submit}>Mark Attendance</button>
      <h3>Detected: {name}</h3>
    </div>
  );
}

export default Dashboard;
