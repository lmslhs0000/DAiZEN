from datetime import timedelta

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.crud.user import authenticate_user


class AuthService:
    @staticmethod
    def login(
        db: Session,
        email: str,
        password: str,
    ) -> str | None:
        user = authenticate_user(
            db=db,
            email=email,
            password=password,
        )

        if user is None:
            return None

        access_token = create_access_token(
            data={"sub": str(user.id)},
            expires_delta=timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
            ),
        )

        return access_token