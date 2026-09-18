import streamlit as st
from langchain.vectorstores import Astra
from langchain_core.tools import tool
from typing import Annotated, Union, Literal
from langchain.agents import create_agent   
from langchain_groq import ChatGroq 
from langsmith import Client
import os 
import dotenv 

dotenv.load_dotenv()