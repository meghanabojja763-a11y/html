from sqlalchemy import Column, Integer, String, Date, Time
from database import Base

class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    date = Column(String)
    time = Column(String)
