from flask import Blueprint, request, redirect, url_for, session
from utils import JSONRes, ResType
from models import User